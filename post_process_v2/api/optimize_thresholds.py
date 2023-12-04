from post_process_v2.base.helpers import dataframe_yield_as_dict_rows
from post_process_v2.api.with_gt import YoloV5WithGTPostProcessor
from post_process_v2.procedures.init_parameters import Threshold
from post_process_v2.procedures.report_ratio import PostProcessReportRatio
from post_process_v2.procedures.analysis_data import PostProcessAnalysisData

from typing import Tuple, List, Dict
from skopt import gp_minimize
from skopt.space import Integer as IntegerSpace
import os.path
import logging
import re


class DefectMetric:
    F_SCORE_BETA = 4

    def __init__(self, tp: int, fn: int, fp: int, overkill_percent: float):
        self.tp = tp
        self.fn = fn
        self.fp = fp
        self.overkill_percent = overkill_percent

    def f1_score(self) -> float:
        precision = self.tp / ((self.tp + self.fp) or 1)
        recall = self.tp / ((self.tp + self.fn) or 1)
        return (1 + self.F_SCORE_BETA) * precision * recall / ((self.F_SCORE_BETA * precision + recall) or 1)

    def value(self, overkill_weight: float):
        assert 0.0 < overkill_weight < 1.0
        logging.debug(f"(1 - f1-score)={1 - self.f1_score():.5f}, overkill={self.overkill_percent:.5f}")
        return (1 - overkill_weight) * (1 - self.f1_score()) + overkill_weight * self.overkill_percent


def build_threshold(
        pmbdb2pmbdbdl_config: Dict[str, Threshold],
        key_list: List[Tuple[str, int]], ratio_list: List[float],
) -> Dict[str, Threshold]:
    # 复制 Threshold
    result = {x: y.copy() for x, y in pmbdb2pmbdbdl_config.items()}
    # 重新设置 min_confidence
    for [name, label_index], ratio in zip(key_list, ratio_list):
        label_index = int(label_index)
        result[name].min_confidence[label_index] = ratio * result[name].min_confidence[label_index]
    return result


class YoloV5ThresholdOptimizer(YoloV5WithGTPostProcessor):
    RATIO_FIELDS = (
        # 阶段,缺陷类型,忽略缺陷,是否包含非重要缺陷,GT,pred标签,pred框
        None, "defect_name", None, None, "gt_count", "pred_count", "pred_distinct_count",
        # 一致数量,一致率,漏检数量,漏检率,错检数量,错检率,过杀数量,过杀率
        "tp", None, "fn", None, "fp", None, None, "overkill_percent",
    )
    METRIC_FIELDS = ("tp", "fn", "fp", "overkill_percent")
    PERCENT_PAT = re.compile(r"([\d.]+)%")
    THRESHOLD_NAMES = ("op2p_parameters", "pmbdb2pmbdbdl_parameters")

    def __init__(
            self, label_id_name_dict: Dict[int, str], p2pmb_thresholds: Dict[str, float],
            pass_label_name: str = None, pass_override_min: float = 0.75,
            overkill_weight: float = 0.14, sample_times: int = 50, max_ratio: int = 20,
    ):
        """
        自动调整 min_confidence 阈值。
        - 将每个 (defect_name, label_index) 的 min_confidence 作为一个待搜索的参数
        - 搜索的范围：以给定的 min_confidence 为中心，在一定半径范围内进行搜索

        :param label_id_name_dict: 全部标签的 `ID-名称` 字典
        :param p2pmb_thresholds: 用于 IoU 过滤的阈值，一般进行调整
        :param pass_label_name: Pass 标签的名称，可以为空。

        :param overkill_weight: 搜索配置参数。用于平衡 `正确率(F1-score)` 与 `过杀率` 的超参，建议小于 1/6
        :param sample_times: 搜索配置参数。搜索次数，前 10 次进行采样，后续次数进行贝叶斯搜索和优化
        :param max_ratio: 搜索配置参数。min_confidence 的最大放大倍数
        """
        YoloV5WithGTPostProcessor.__init__(
            self, label_id_name_dict, p2pmb_thresholds, pass_label_name, pass_override_min,
        )
        self.ratio_field_dict = dict(zip(PostProcessReportRatio.EXPORT_FIELD_NAMES, self.RATIO_FIELDS))
        # 设置 `搜索超参`
        self.overkill_weight = overkill_weight
        self.sample_times = sample_times
        self.max_ratio = max_ratio
        # 校验 `搜索超参`
        self.__verify_hyper_parameters()

    '''
    private
    '''

    def __verify_hyper_parameters(self) -> bool:
        conditions = [
            0.0 < self.overkill_weight < 1.0,
            self.sample_times >= 20,
            self.max_ratio >= 2,
        ]
        error_messages = [
            "overkill_weight 应在 (0.0, 1.0) 之间",
            "sample_times 应大于等于 20",
            "max_ratio 应大于等于 2",
        ]
        result = True
        for condition, message in zip(conditions, error_messages):
            if not condition:
                logging.error(message)
            result = result and condition
        if not result:
            raise ValueError("搜索超参数非法")
        return result

    def __get_and_parse_ratios(self) -> Dict[str, Dict]:
        # 获取 ratio 指标
        ratio_data = self.process_list[-1][1].result
        # 解析行数据
        result = dict()
        for row in dataframe_yield_as_dict_rows(ratio_data):
            # key 值替换
            row = {self.ratio_field_dict[x]: y for x, y in row.items() if self.ratio_field_dict[x]}
            # 类型转换
            if not row["overkill_percent"]:
                row["overkill_percent"] = 0.0
            else:
                row["overkill_percent"] = float(self.PERCENT_PAT.search(row["overkill_percent"]).group(1)) * 0.01
            result[row["defect_name"]] = row
        return result

    def __calc_weighted_mean_metric(self, ratio_dict: Dict[str, Dict]):
        weight_sum = 0
        result = 0.0
        for _id, _name in self.label_id_name_dict.items():
            if _name == self.pass_label_name:
                continue
            # 计算指标和权重
            _metric_data = {x: y for x, y in ratio_dict[_name].items() if x in self.METRIC_FIELDS}
            _weight = ratio_dict[_name]["gt_count"]
            # 累加
            weight_sum += _weight
            result += _weight * DefectMetric(**_metric_data).value(self.overkill_weight)
        result /= (weight_sum or 1)
        return result

    '''
    protected
    '''

    def _define_search_space(self, threshold_dict: Dict[str, Threshold]) \
            -> Tuple[List[Tuple[str, int]], List[IntegerSpace]]:
        key_list, space_list = [], []
        for name, threshold in threshold_dict.items():
            if name == self.pass_label_name:
                continue
            for label_index, confidence in threshold.min_confidence.items():
                _max = [i for i in range(1, 1 + self.max_ratio) if confidence * i < 1.0][-1]
                if _max <= 1:
                    continue
                key_list.append((name, label_index))
                space_list.append(IntegerSpace(low=1, high=_max))
        return key_list, space_list

    def _run_once(self, op2p_config: Dict[str, Threshold], pmbdb2pmbdbdl_config: Dict[str, Threshold]) -> float:
        # 执行后处理
        self.process_list = self._op_processes(op2p_config, pmbdb2pmbdbdl_config)
        p_data_list, map_data_list = self._build_p_map_gt_data(self.process_list)
        self._calc_ratio(p_data_list[-1:], map_data_list[-1:])
        # 获取 ratio 指标
        ratio_dict = self.__get_and_parse_ratios()
        logging.debug(f"r4ratio: {ratio_dict['所有']}")
        # 计算综合指标(越小越好)
        result = self.__calc_weighted_mean_metric(ratio_dict)
        result *= 100
        logging.debug(f"metric: {result}")
        return result

    def _build_run_func(
            self, op2p_config: Dict[str, Threshold], pmbdb2pmbdbdl_config: Dict[str, Threshold],
            key_list: List[Tuple[str, int]],
    ):
        def func(ratio_list: List[float]) -> float:
            logging.debug(f"ratio-list: {ratio_list}")
            # 构造阈值字典
            threshold_dict = build_threshold(pmbdb2pmbdbdl_config, key_list, ratio_list)
            # 执行后处理
            return self._run_once(op2p_config, threshold_dict)

        return func

    '''
    public
    '''

    def optimize(
            self, op2p_config: Dict[str, Threshold],
            pmbdb2pmbdbdl_config: Dict[str, Threshold], dest_dir: str = None,
    ) -> Tuple[Dict[str, Threshold], Dict[str, Threshold]]:
        logging.info(f"优化前的综合指标: {self._run_once(op2p_config, pmbdb2pmbdbdl_config)}")
        # 定义搜索空间
        key_list, search_space = self._define_search_space(pmbdb2pmbdbdl_config)
        # 贝叶斯搜索: 不包括 pass 标签的阈值
        result = gp_minimize(
            func=self._build_run_func(op2p_config, pmbdb2pmbdbdl_config, key_list),
            dimensions=search_space, acq_func="gp_hedge",
            n_calls=self.sample_times, verbose=False, n_jobs=1,
        )
        # 获取搜索结果
        new_pmbdb2pmbdbdl_config = build_threshold(pmbdb2pmbdbdl_config, key_list, result.x)
        logging.info(f"优化后，最优的放大倍率: {result.x}")
        logging.info(f"优化后的综合指标: {self._run_once(op2p_config, new_pmbdb2pmbdbdl_config)}")
        logging.info(f"优化后的 r4ratio: {self.__get_and_parse_ratios()['所有']}")
        # 导出
        if dest_dir:
            data_list = (op2p_config, new_pmbdb2pmbdbdl_config)
            for name, data in zip(self.THRESHOLD_NAMES, data_list):
                PostProcessAnalysisData.export_one_threshold(data, os.path.join(dest_dir, f"{name}.json"))
        # 返回结果
        return op2p_config, new_pmbdb2pmbdbdl_config

    def optimize_with_parameters_from_dir(self, src_folder: str, dest_folder: str = None):
        # 指定的参数文件夹下，应包含以下两个文件:
        # op2p_parameters.json
        # pmbdb2pmbdbdl_parameters.json
        return self.optimize(
            dest_dir=dest_folder,
            **self._load_parameter_dir_files(src_folder),
        )
