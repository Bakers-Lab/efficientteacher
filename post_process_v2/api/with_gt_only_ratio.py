from post_process_v2.api.components.with_gt_import_export_trait import WithGTImportExportTrait
from post_process_v2.api.base import YoloV5PostProcessorBase
from post_process_v2.procedures.match_with_gt import PostProcessMatchWithGT
from post_process_v2.procedures.report_ratio import PostProcessReportRatio
from post_process_v2.base.helpers import dataframe_group_by

from typing import Tuple, List, Iterator, Dict
import pandas as pd
import logging
import time


class YoloV5WithGTOnlyRatio(WithGTImportExportTrait, YoloV5PostProcessorBase):
    OP_LABEL = "PMBDBDL_FILTERED"

    DEFAULT_OP_TABLE_NAME = "PMBDBDL_FILTERED"
    # ["ID", "CenterX", "CenterY", "Length", "Width", "Confidence", "ImageID", "LableID", "Area", "BoxID", "LableIndex"]
    DEFAULT_OP_COLUMNS = ["CenterX", "CenterY", "Length", "Width", "ImageID", "LableID", "Confidence", "LableIndex"]

    def __init__(
            self, label_id_name_dict: Dict[int, str],
            pass_label_name: str = None, pass_override_min: float = 0.75,
    ):
        WithGTImportExportTrait.__init__(self)
        YoloV5PostProcessorBase.__init__(
            self, label_id_name_dict, p2pmb_thresholds=dict(),
            pass_label_name=pass_label_name, pass_override_min=pass_override_min,
        )
        # values
        self.gt_data = None

    '''
    protected 方法
    '''

    def _reset_op_data(self) -> pd.DataFrame:
        op_data = self.op_data.copy()
        # Area 属性
        op_data["Area"] = op_data["Length"] * op_data["Width"]
        op_data = op_data[op_data["Area"] > 0.0]
        # BoxID 属性
        op_data = op_data.sort_values(by=["ImageID", "CenterX", "CenterY", "Width", "Length"])
        result = []
        for image_boxes in dataframe_group_by(op_data, by=["ImageID"]).values():
            location_set = set()
            for box_row in image_boxes:
                location_set.add((box_row["CenterX"], box_row["CenterY"], box_row["Width"], box_row["Length"]))
                box_row["BoxID"] = len(location_set)
                result.append(box_row)
        # 格式转换
        result = pd.DataFrame(result)
        return result

    def _calc_ratio_directly(self):
        # 构造数据变量
        p_data_list = [(self.OP_LABEL, self._reset_op_data())]
        gt_map_data_list = [
            (_name, PostProcessMatchWithGT(self.gt_data, _data).run(self.pass_label_id))
            for _name, _data in p_data_list
        ]
        # 计算 ratio
        self.process_list.append((
            "R4Ratio",
            PostProcessReportRatio(self.label_id_name_dict, self.pass_label_name, p_data_list, gt_map_data_list).run(),
        ))
        logging.info("指标计算结束")

    def _input_data_for_export(self) -> Iterator[Tuple[str, pd.DataFrame, List[str]]]:
        name_list = [self.OP_LABEL, "GT"]
        data_list = [self._reset_op_data(), self.gt_data]
        columns_list = [self.DEFAULT_OP_COLUMNS, self.DEFAULT_GT_COLUMNS]
        return zip(name_list, data_list, columns_list)

    '''
    public: 入口方法
    '''

    def run(self):
        self._verify_dataframe_columns()
        # 开始处理
        time_start = time.time()
        logging.info("[start] 开始: 直接根据 PMBDBDL 数据计算 ratio")
        self._calc_ratio_directly()
        logging.info(f"[end] 结束: 直接根据 PMBDBDL 数据计算 ratio，耗时 {time.time() - time_start:.2f}s")
        return self

    '''
    public: 输入
    '''

    # 父类方法: load_from_numpy

    # 父类方法: load_from_dataframe

    # 父类方法: load_from_csv

    # 父类方法: load_from_db

    '''
    public: 输出
    '''

    # 父类方法: export_to_csv

    # 父类方法: export_to_db

    # 父类方法: export_ratio_to_dataframe
