from post_process_v2.base.helpers import group_dict_list_by
from post_process_v2.procedures.components.calc_metrics import MetricsCalculater

from typing import List, Dict
import pandas as pd


class WithPassLabelMetricsCalculater(MetricsCalculater):

    def __init__(
            self, id_name_dict: dict, pass_label_name: str,
            gt_map_data: pd.DataFrame, p_data: pd.DataFrame
    ):
        MetricsCalculater.__init__(self, id_name_dict, gt_map_data, p_data)
        # Pass 标签
        self.pass_label_name = pass_label_name
        self.pass_label_id = self.name_id_dict[pass_label_name]
        self.pass_label_box_count = self.__calc_pass_label_box_count()
        self.pass_match_count = self.__calc_pass_match_count()

    '''
    private
    '''

    def __calc_pass_label_box_count(self):
        result = 0
        for _image_id, _map_rows in self.gt_map_image_dict.items():
            _image_defect_p_rows = [
                x for x in _map_rows
                if not pd.isna(x["p_label_id"]) and x["p_label_id"] != self.pass_label_id
            ]
            # 要求: 图中 p-box 的标签均不是 defect-label
            if _image_defect_p_rows:
                continue
            # 对每个 gt-box，均看作存在一个 pass 标签的 p-box 与之对应
            for _gt_box_id, _rows in group_dict_list_by(_map_rows, "gt_box_id").items():
                result += 1
        return result

    def __calc_pass_match_count(self):
        result = 0
        for _image_id, _map_rows in self.gt_map_image_dict.items():
            _image_defect_p_rows = [
                x for x in _map_rows
                if not pd.isna(x["p_label_id"]) and x["p_label_id"] != self.pass_label_id
            ]
            # 要求: 图中 p-box 的标签均不是 defect-label
            if _image_defect_p_rows:
                continue
            # 对每个 gt-box，均看作存在一个 pass 标签的 p-box 与之对应
            for _gt_box_id, _rows in group_dict_list_by(_map_rows, "gt_box_id").items():
                if pd.isna(_rows[0]["gt_label_id"]):
                    continue
                # 当 gt-label 也为 pass 标签时，看作 `Pass 一致`
                result += 1 if _rows[0]["gt_label_id"] == self.pass_label_id else 0
        return result

    '''
    protected: 计算 metrics
    '''

    def _calc_basic_metrics(self, label_ids: list) -> Dict[str, int]:
        defect_label_ids = [x for x in label_ids if x != self.pass_label_id]
        # 数量统计
        p_box_ids = [
            (x["ImageID"], x["BoxID"]) for _rows in self.p_image_dict.values()
            for x in _rows if x["LableID"] in defect_label_ids
        ]
        gt_box_ids = [
            (x["image_id"], x["gt_box_id"]) for _rows in self.gt_map_image_dict.values()
            for x in _rows if not pd.isna(x["gt_label_id"]) and x["gt_label_id"] in label_ids
        ]
        result = {
            "gt_box_count": len(set(gt_box_ids)), "p_box_count": len(p_box_ids),
            "p_box_distinct_count": len(set(p_box_ids)),
        }
        # 基于 gt_data 计算: Missing, Match, Wrong
        result.update({x: 0 for x in self.METRIC_NAME_FUNC_DICT})
        for _map_rows in self.gt_map_image_dict.values():
            _map_rows = [x for x in _map_rows if not pd.isna(x["gt_label_id"]) and x["gt_label_id"] in defect_label_ids]
            for _rows in group_dict_list_by(_map_rows, "gt_box_id").values():
                for _key, _func in self.METRIC_NAME_FUNC_DICT.items():
                    result[_key] += 1 if _func(_rows) else 0
        # Pass 标签修正
        if self.pass_label_id in label_ids:
            result["p_box_count"] += self.pass_label_box_count
            result["p_box_distinct_count"] += self.pass_label_box_count
            result["Match"] += self.pass_match_count
        # 基于 gt_map_data 计算: OverKill
        result["OverKill"] = 0
        for _map_rows in self.gt_map_image_dict.values():
            # 情况一: 无 gt-box 与 p-box 相匹配
            result["OverKill"] += len([
                x for x in _map_rows
                if pd.isna(x["gt_label_id"]) and x["p_label_id"] in defect_label_ids
            ])
            # 情况二: 与 p-box 匹配的 gt-box 的标签为 Pass 标签
            for _row in _map_rows:
                # 存在与 p-box 匹配的 gt-box
                if pd.isna(_row["gt_label_id"]) or pd.isna(_row["p_label_id"]):
                    continue
                # g-label == pass & p-label != pass
                if _row["gt_label_id"] == self.pass_label_id and _row["p_label_id"] in defect_label_ids:
                    result["OverKill"] += 1
        # 返回结果
        return result

    '''
    protected: 构建行数据
    '''

    def _parameters_to_description(self, label_ids, black_label_names, pass_label_names) -> Dict[str, str]:
        label_names = [self.id_name_dict[x] for x in label_ids]
        return {
            "缺陷类型": "所有" if len(label_ids) == len(self.id_name_dict) else "+".join(label_names),
            "忽略缺陷": "无" if not black_label_names else f"不包含: {'+'.join(black_label_names)}",
            "是否包含非重要缺陷": "无忽略缺陷" if not pass_label_names else f"PASS 标签: {'+'.join(pass_label_names)}",
        }

    '''
    public
    '''

    def run(self, white_label_names: List[str] = None, black_label_names: List[str] = None) -> Dict:
        if not ((white_label_names is None) ^ (black_label_names is None)):
            # 如果需要使用全部 label, 设置 black_label_names = [] 即可
            raise ValueError("label_name 白名单和黑名单，必须且只能指定其中一个(不是 None)")
        # 参数处理
        label_ids = self._calc_label_ids(white_label_names, black_label_names)
        # 计算 Match, Missing, Wrong, OverKill
        metrics = self._calc_basic_metrics(label_ids)
        # 构建行数据
        result = self._parameters_to_description(label_ids, black_label_names, [self.pass_label_name])
        result.update(self._metrics_to_description(metrics))
        return result
