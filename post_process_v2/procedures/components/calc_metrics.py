from post_process_v2.base.helpers import dataframe_group_by, group_dict_list_by

from typing import List, Dict
import pandas as pd


def is_p_box_missing(map_rows: List[Dict]) -> bool:
    return len(map_rows) == 1 and pd.isna(map_rows[0]["p_box_id"])


def is_p_box_matched(map_rows: List[Dict]) -> bool:
    map_rows = [x for x in map_rows if not pd.isna(x["p_label_id"]) and x["gt_label_id"] == x["p_label_id"]]
    return len(map_rows) > 0


def is_p_box_wrong(map_rows: List[Dict]) -> bool:
    # 注意: p_label_ids 为空时，标记为 Missing 而非 Wrong
    if is_p_box_missing(map_rows):
        return False
    map_rows = [x for x in map_rows if x["gt_label_id"] == x["p_label_id"]]
    return len(map_rows) == 0


class MetricsCalculater:
    TO_CORRECT_METRIC_KEYS = ("Missing", "Wrong", "Overkill")
    METRIC_NAME_FUNC_DICT = {
        "Missing": is_p_box_missing,
        "Match": is_p_box_matched,
        "Wrong": is_p_box_wrong,
    }

    METRIC_COUNT_KEY_DICT = {
        "gt_box_count": "GT", "p_box_count": "pred标签",
        "p_box_distinct_count": "pred框",
    }
    METRIC_PERF_KEY_DICT = {
        "Missing": "漏检", "Match": "一致", "Wrong": "错检", "OverKill": "过杀",
    }

    METRIC_PERF_PERCENT_BASE_DICT = {
        "Missing": "gt_box_count", "Match": "gt_box_count",
        "Wrong": "gt_box_count", "OverKill": "p_box_count",
    }

    def __init__(
            self, id_name_dict: dict,
            gt_map_data: pd.DataFrame, p_data: pd.DataFrame
    ):
        # params
        self.gt_map_data = gt_map_data
        # 分类标签信息
        self.id_name_dict = id_name_dict
        self.name_id_dict = {y: x for x, y in id_name_dict.items()}
        # data
        self.p_image_dict = dataframe_group_by(p_data, by="ImageID")
        self.gt_map_image_dict = dataframe_group_by(gt_map_data, by="image_id")

    def _calc_label_ids(self, white_label_names, black_label_names) -> List[int]:
        if white_label_names:
            result = white_label_names
        else:
            result = [x for x in self.name_id_dict.keys() if x not in black_label_names]
        result = [self.name_id_dict[x] for x in result]
        return result

    '''
    protected: 计算 metrics
    '''

    def _calc_basic_metrics(self, label_ids: list) -> Dict[str, int]:
        # 数量统计
        p_box_ids = [
            (x["ImageID"], x["BoxID"]) for _rows in self.p_image_dict.values()
            for x in _rows if x["LableID"] in label_ids
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
            _map_rows = [x for x in _map_rows if not pd.isna(x["gt_label_id"]) and x["gt_label_id"] in label_ids]
            for _rows in group_dict_list_by(_map_rows, "gt_box_id").values():
                for _key, _func in self.METRIC_NAME_FUNC_DICT.items():
                    result[_key] += 1 if _func(_rows) else 0
        # 基于 gt_map_data 计算: OverKill
        result["OverKill"] = 0
        for _map_rows in self.gt_map_image_dict.values():
            result["OverKill"] += len([
                x for x in _map_rows
                if pd.isna(x["gt_label_id"]) and x["p_label_id"] in label_ids
            ])
        # 返回结果
        return result

    def _correct_metrics_with(self, metrics: dict, pass_label_ids: list) -> Dict[str, int]:
        result = metrics.copy()
        if not metrics:
            return result
        metrics_diff = self._calc_basic_metrics(pass_label_ids)
        # 修正
        for _key in self.TO_CORRECT_METRIC_KEYS:
            result[_key] -= metrics_diff[_key]
        result["Match"] = metrics_diff["Wrong"] + metrics_diff["Missing"]
        return result

    '''
    protected: 构建行数据
    '''

    def _parameters_to_description(self, label_ids, black_label_names, pass_label_names) -> Dict[str, str]:
        label_names = [self.id_name_dict[x] for x in label_ids]
        return {
            "缺陷类型": "所有" if len(label_ids) == len(self.id_name_dict) else "+".join(label_names),
            "忽略缺陷": "无" if not black_label_names else f"不包含: {'+'.join(black_label_names)}",
            "是否包含非重要缺陷": "无忽略缺陷" if not pass_label_names else f"{'+'.join(pass_label_names)}",
        }

    def _metrics_to_description(self, metrics: dict) -> Dict[str, str]:
        # 数量的统计信息
        result = {y: metrics[x] for x, y in self.METRIC_COUNT_KEY_DICT.items()}
        # performance-metric
        result.update({f"{y}数量": metrics[x] for x, y in self.METRIC_PERF_KEY_DICT.items()})
        # performance-metric 的比率值
        for performance_key, count_key in self.METRIC_PERF_PERCENT_BASE_DICT.items():
            _key = f"{self.METRIC_PERF_KEY_DICT[performance_key]}率"
            if not metrics[count_key]:
                _value = None
            else:
                _value = f"{round(100 * metrics[performance_key] / metrics[count_key], 2)}%"
            result[_key] = _value
        return result

    '''
    public
    '''

    def run(self, white_label_names=None, black_label_names=None) -> Dict:
        if not ((white_label_names is None) ^ (black_label_names is None)):
            # 如果需要使用全部 label, 设置 black_label_names = [] 即可
            raise ValueError("label_name 白名单和黑名单，必须且只能指定其中一个(不是 None)")
        # 参数处理
        label_ids = self._calc_label_ids(white_label_names, black_label_names)
        # 计算 Match, Missing, Wrong, OverKill
        metrics = self._calc_basic_metrics(label_ids)
        # 构建行数据
        result = self._parameters_to_description(label_ids, black_label_names, [])
        result.update(self._metrics_to_description(metrics))
        return result
