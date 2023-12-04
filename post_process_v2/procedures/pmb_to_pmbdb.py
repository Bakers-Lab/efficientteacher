from post_process_v2.base.helpers import method_run_once_limit, dataframe_group_by
from post_process_v2.procedures.components.base_p_procedure import BaseProcedure

import pandas as pd


class PostProcessPMB2PMBDB(BaseProcedure):
    EXPORT_FIELD_NAMES = [
        "ID", "CenterX", "CenterY", "Length", "Width", "Confidence",
        "ImageID", "LableID", "Area", "BoxID", "LableIndex",
    ]

    def __init__(self, data: pd.DataFrame):
        BaseProcedure.__init__(self, data.copy())

    '''
    public
    '''

    @method_run_once_limit
    def run(self):
        self.data = self.data.sort_values(by=["ImageID", "BoxID", "Confidence", "LableID"], ascending=True)
        self.data = self.data.reset_index(drop=True)
        # 添加主键
        self.data["ID"] = self.data.index
        # 分组，并记录组内排名
        result = []
        for _, image_boxes in dataframe_group_by(self.data, by=["ImageID", "BoxID"]).items():
            image_boxes.sort(key=lambda x: (x["Confidence"], x["ID"]), reverse=True)
            for i, _row in enumerate(image_boxes):
                _row["LableIndex"] = i + 1
                result.append(_row)
        self._result = pd.DataFrame(result)
        return self
