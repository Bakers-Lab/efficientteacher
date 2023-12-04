from post_process_v2.procedures.op_to_p import PostProcessOP2P
from post_process_v2.procedures.init_parameters import Threshold
from post_process_v2.base.helpers import method_run_once_limit, dataframe_group_by

from typing import Dict, List
import pandas as pd


class PostProcessPMBDB2PMBDBDL(PostProcessOP2P):
    EXPORT_FIELD_NAMES = [
        "ID", "CenterX", "CenterY", "Length", "Width", "Confidence",
        "ImageID", "LableID", "Area", "BoxID", "LableIndex",
    ]

    def __init__(
            self, data: pd.DataFrame, thresholds: Dict[str, Threshold],
            pass_label_id: int, pass_override_min: float,
    ):
        PostProcessOP2P.__init__(self, data, thresholds)
        self.pass_label_id = pass_label_id
        self.pass_override_min = pass_override_min

    def _build_export_data(self) -> pd.DataFrame:
        return self._result.sort_values(by="ID")

    def _filter_pass_label_boxes(self, box_rows: List[Dict]) -> List[Dict]:
        box_rows.sort(key=lambda x: x["LableIndex"])
        # 无 pass 标签信息 | 无效的过滤条件
        if self.pass_label_id is None or self.pass_override_min >= 1.0:
            return box_rows
        # 要求 Pass 标签的置信度是最高的
        if not box_rows or box_rows[0]["LableID"] != self.pass_label_id:
            return box_rows
        # 当置信度之间差值大于 pass_override_min 时，忽略该 label
        result = [box_rows[0]]
        for row in box_rows[1:]:
            if box_rows[0]["Confidence"] - row["Confidence"] >= self.pass_override_min:
                continue
            result.append(row)
        return result

    @method_run_once_limit
    def run(self):
        result = []
        # 使用阈值筛选
        for key, box_rows in dataframe_group_by(self.data, by=["ImageID", "BoxID"]).items():
            # Pass 标签覆盖规则
            box_rows = self._filter_pass_label_boxes(box_rows)
            # 按给定阈值下限过滤
            for _row in box_rows:
                if not self.threshold_dict[_row["LableID"]].is_row_ok(_row):
                    continue
                result.append(_row)
        self._result = pd.DataFrame(result)
        return self
