from post_process_v2.base.helpers import method_run_once_limit, dataframe_yield_as_dict_rows
from post_process_v2.procedures.components.base_p_procedure import BaseProcedure
from post_process_v2.procedures.init_parameters import Threshold

from typing import Dict
import pandas as pd


class PostProcessOP2P(BaseProcedure):
    EXPORT_FIELD_NAMES = [
        "CenterX", "CenterY", "Length", "Width", "Confidence", "ImageID", "LableID", "Area", "BoxID",
    ]

    def __init__(self, data: pd.DataFrame, thresholds: Dict[str, Threshold]):
        BaseProcedure.__init__(self, data)
        self.threshold_dict = {y.label_id: y for x, y in thresholds.items()}

    @method_run_once_limit
    def run(self):
        result = []
        # 使用阈值筛选
        for _row in dataframe_yield_as_dict_rows(self.data):
            if not self.threshold_dict[_row["LableID"]].is_row_ok(_row):
                continue
            result.append(_row)
        self._result = pd.DataFrame(result)
        return self
