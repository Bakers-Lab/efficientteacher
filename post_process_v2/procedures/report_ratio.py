from post_process_v2.base.helpers import method_run_once_limit
from post_process_v2.procedures.components.calc_metrics import MetricsCalculater
from post_process_v2.procedures.components.calc_metrics_with_pass_label import WithPassLabelMetricsCalculater
from post_process_v2.procedures.components.base_p_procedure import BaseProcedure

from typing import List, Tuple
import pandas as pd


class PostProcessReportRatio(BaseProcedure):
    EXPORT_FIELD_NAMES = [
        "阶段", "缺陷类型", "忽略缺陷", "是否包含非重要缺陷",
        "GT", "pred标签", "pred框",
        "一致数量", "一致率", "漏检数量", "漏检率", "错检数量", "错检率", "过杀数量", "过杀率",
    ]

    COMMON_CONDITIONS = [
        # [white_label_names, black_label_names]
        # 全部标签
        [None, []],
        # 全部标签（不包含 other 标签）
        [None, ["other"]],
    ]

    def __init__(
            self, id_name_dict: dict, pass_label_name: str,
            p_data_list: List[Tuple[str, pd.DataFrame]],
            gt_data_list: List[Tuple[str, pd.DataFrame]],
    ):
        if len(p_data_list) != len(gt_data_list):
            raise ValueError("p 数据列和 gt 数据列的的长度应保持一致")
        BaseProcedure.__init__(self)
        self.id_name_dict = id_name_dict
        self.pass_label_name = pass_label_name
        self.gt_data_list = gt_data_list
        self.p_data_list = p_data_list

    '''
    protected
    '''

    def _one_process_step(
            self, gt_name: str, gt_data: pd.DataFrame,
            p_name: str, p_data: pd.DataFrame,
    ):
        if gt_name != p_name:
            raise ValueError("只允许处理来自同一 process step 对应的数据")
        if self.pass_label_name:
            calculater = WithPassLabelMetricsCalculater(
                self.id_name_dict, self.pass_label_name, gt_data, p_data,
            )
        else:
            calculater = MetricsCalculater(self.id_name_dict, gt_data, p_data)
        result = []
        # 多标签条件下计算
        for _condition in self.COMMON_CONDITIONS:
            result.append(calculater.run(*_condition))
        # 单个标签条件下计算
        for _name in self.id_name_dict.values():
            result.append(calculater.run(white_label_names=[_name]))
        # 构造行数据
        for _row in result:
            _row["阶段"] = gt_name
        return result

    '''
    public
    '''

    @method_run_once_limit
    def run(self):
        result = []
        for gt_info, p_info in zip(self.gt_data_list, self.p_data_list):
            result.extend(self._one_process_step(*gt_info, *p_info))
        result = pd.DataFrame(result)
        self._result = result
        return self
