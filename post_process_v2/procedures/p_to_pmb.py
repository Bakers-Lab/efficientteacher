from post_process_v2.base.helpers import dataframe_group_by, method_run_once_limit
from post_process_v2.procedures.components.base_p_procedure import BaseProcedure
from post_process_v2.procedures.components.bbox_merger import BoundBoxMerger, BBox

import pandas as pd


class PostProcessP2PMB(BaseProcedure):
    EXPORT_FIELD_NAMES = [
        "CenterX", "CenterY", "Length", "Width", "Confidence", "ImageID", "LableID", "Area", "BoxID",
    ]

    def __init__(self, data: pd.DataFrame, **thresholds):
        BaseProcedure.__init__(self, data)
        self.thresholds = dict(thresholds)

    '''
    public
    '''

    @method_run_once_limit
    def run(self):
        result = []
        # 逐个处理每张图片
        for image_id, boxes in dataframe_group_by(self.data, "ImageID").items():
            bbox_list = [BBox.load_from_dict(x) for x in boxes]
            result.extend(BoundBoxMerger(bbox_list, **self.thresholds).run())
        # 构造处理结果: 去除不必要的字段
        self._result = pd.DataFrame(
            data=[{k: row[k] for k in self.EXPORT_FIELD_NAMES} for row in result],
            columns=self.EXPORT_FIELD_NAMES,
        )
        return self
