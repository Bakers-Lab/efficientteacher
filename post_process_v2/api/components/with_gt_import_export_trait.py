from post_process_v2.api.base import YoloV5PostProcessorBase
from post_process_v2.base.helpers import read_table_as_dataframe, triple_line_logging
from post_process_v2.base.exceptions import *
from post_process_v2.mysql.reader import Reader
from post_process_v2.procedures.report_ratio import PostProcessReportRatio

from typing import Union
import pandas as pd
import numpy as np


class WithGTImportExportTrait:
    DEFAULT_OP_TABLE_NAME = YoloV5PostProcessorBase.DEFAULT_OP_TABLE_NAME
    DEFAULT_GT_TABLE_NAME = YoloV5PostProcessorBase.DEFAULT_GT_TABLE_NAME
    DEFAULT_GT_COLUMNS = YoloV5PostProcessorBase.DEFAULT_GT_COLUMNS
    DEFAULT_OP_COLUMNS = YoloV5PostProcessorBase.DEFAULT_OP_COLUMNS

    def __init__(self):
        self.op_data = None
        self.gt_data = None
        self.process_list = []

    '''
    protected: 校验方法
    '''

    def _verify_dataframe_columns(self) -> bool:
        # 确保数据已载入
        if self.op_data is None or self.gt_data is None:
            return False
        # 列顺序可以不一致
        if set(self.op_data.columns) != set(self.DEFAULT_OP_COLUMNS):
            raise ValueError(f"OP 数据的列名不正确: {self.op_data.columns}")
        if set(self.gt_data.columns) != set(self.DEFAULT_GT_COLUMNS):
            raise ValueError(f"GT 数据的列名不正确: {self.gt_data.columns}")
        # 去除掉面积为 0 的预测框
        if "Area" in self.op_data.columns:
            self.op_data = self.op_data[self.op_data["Area"] > 0.0]
        return True

    '''
    public: 输入
    '''

    def load_from_numpy(self, gt_data: np.array, op_data: np.array, gt_cols=None, op_cols=None):
        triple_line_logging("[load] 载入 numpy 格式的数据")
        self.gt_data = pd.DataFrame(gt_data, columns=gt_cols or self.DEFAULT_GT_COLUMNS)
        self.op_data = pd.DataFrame(op_data, columns=op_cols or self.DEFAULT_OP_COLUMNS)
        return self

    def load_from_dataframe(self, gt_data: pd.DataFrame, op_data: pd.DataFrame):
        triple_line_logging("[load] 载入 dataframe 格式的数据")
        self.gt_data = gt_data
        self.op_data = op_data
        return self

    def load_from_csv(
            self, gt_path: str, op_path: str,
            header_mode: Union[str, None] = 'infer', gt_cols=None, op_cols=None,
    ):
        triple_line_logging("[load] 载入 csv 格式的数据")
        self.gt_data = pd.read_csv(gt_path, header=header_mode)
        self.op_data = pd.read_csv(op_path, header=header_mode)
        # 默认 header
        if header_mode is None:
            self.op_data.columns = self.DEFAULT_OP_COLUMNS
            self.gt_data.columns = self.DEFAULT_GT_COLUMNS
        # 覆盖 header
        if gt_cols:
            self.gt_data.columns = gt_cols
        if op_cols:
            self.op_data.columns = op_cols
        return self

    def load_from_db(self, connect_options: dict, op_table: str = None, gt_table: str = None):
        triple_line_logging("[load] 从数据库载入数据")
        reader = Reader(connect_options, silent=True)
        self.op_data = read_table_as_dataframe(reader, op_table or self.DEFAULT_OP_TABLE_NAME)
        self.gt_data = read_table_as_dataframe(reader, gt_table or self.DEFAULT_GT_TABLE_NAME)
        return self

    '''
    public: 输出
    '''

    def export_ratio_to_dataframe(self) -> pd.DataFrame:
        if not self.process_list:
            raise ExportBeforeRunError("导出数据前，需要先执行后处理")
        p_name, p = self.process_list[-1]
        if not isinstance(p, PostProcessReportRatio):
            raise ValueError("后处理过程中，未执行数据分析，无法导出分析结果")
        triple_line_logging("[export] 导出 ratio 数据到 dataframe 变量")
        return p.result.copy()
