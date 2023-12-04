from post_process_v2.base.helpers import dataframe_group_by

from typing import List, Dict, Tuple, Union
import pandas as pd
import numpy as np


def verify_boxes_match_status(box1: dict, box2: dict) -> bool:
    return abs(box1["CenterX"] - box2["CenterX"]) < (box1["Length"] + box2["Length"]) / 2 \
           and abs(box1["CenterY"] - box2["CenterY"]) < (box1["Width"] + box2["Width"]) / 2


def full_join(gt_box_list: List[Dict], p_box_list: List[Dict]) \
        -> List[Tuple[Union[None, Dict], Union[None, Dict]]]:
    # 为行数据编号
    gt_box_dict = dict(zip(range(len(gt_box_list)), gt_box_list))
    p_box_dict = dict(zip(range(len(p_box_list)), p_box_list))
    # INNER JOIN 操作，并使用 id 记录 join 结果
    matched_gt_ids, matched_p_ids = set(), set()
    result = []
    for gt_id, gt_row in gt_box_dict.items():
        for p_id, p_row in p_box_dict.items():
            if not verify_boxes_match_status(gt_row, p_row):
                continue
            matched_gt_ids.add(gt_id)
            matched_p_ids.add(p_id)
            result.append((gt_row, p_row))
    # INNER JOIN -> FULL JOIN
    for gt_id, gt_row in gt_box_dict.items():
        if gt_id not in matched_gt_ids:
            result.append((gt_row, None))
    for p_id, p_row in p_box_dict.items():
        if p_id not in matched_p_ids:
            result.append((None, p_row))
    return result


class PostProcessMatchWithGT:
    RESULT_D_TYPE_DICT = {
        "image_id": "Int64", "p_box_id": "Int64", "gt_box_id": "Int64", "p_label_id": "Int64",
        "gt_label_id": "Int64", "p_label_index": "Int64", "confidence": float,
    }

    def __init__(self, gt_data: pd.DataFrame, p_data: pd.DataFrame):
        self.gt_data = gt_data
        self.p_data = p_data

    '''
    protected
    '''

    def _format_p_row(self, p_box: dict) -> dict:
        return {
            "p_box_id": p_box["BoxID"], "p_label_id": p_box["LableID"],
            "confidence": p_box["Confidence"],
            "p_label_index": p_box["LableIndex"] if "LableIndex" in self.p_data.columns else None,
        }

    def _format_gt_row(self, gt_box: dict) -> dict:
        return {"gt_box_id": gt_box["BoxID"], "gt_label_id": gt_box["LableID"]}

    def _empty_p_row(self):
        return {
            "p_box_id": np.nan,
            "confidence": np.nan,
            "p_label_index": np.nan,
            "p_label_id": np.nan,
        }

    def _empty_gt_row(self):
        return {"gt_box_id": np.nan, "gt_label_id": np.nan}

    '''
    public
    '''

    def run(self, pass_label_id: int = None) -> pd.DataFrame:
        # 数据预处理
        p_image_dict = dataframe_group_by(
            self.p_data, "ImageID",
            filter_func=lambda x: pass_label_id is None or pass_label_id != x["LableID"],
        )
        gt_image_dict = dataframe_group_by(self.gt_data, "ImageID")
        # 匹配 gt-box 与 p-box
        result = []
        for image_id, gt_rows in gt_image_dict.items():
            op_rows = p_image_dict.get(image_id, [])
            # FULL JOIN 操作
            for gt_box, p_box in full_join(gt_rows, op_rows):
                map_row = {"image_id": image_id}
                map_row.update(self._format_gt_row(gt_box) if gt_box else self._empty_gt_row())
                map_row.update(self._format_p_row(p_box) if p_box else self._empty_p_row())
                result.append(map_row)
        # 返回匹配结果
        return pd.DataFrame(result).astype(self.RESULT_D_TYPE_DICT)
