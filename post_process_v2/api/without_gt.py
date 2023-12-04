from post_process_v2.api.base import YoloV5PostProcessorBase
from post_process_v2.base.helpers import read_table_as_dataframe, triple_line_logging
from post_process_v2.base.exceptions import ExportBeforeRunError
from post_process_v2.mysql.reader import Reader
from post_process_v2.procedures.init_parameters import Threshold
from post_process_v2.procedures.pmbdb_to_pmbdbdl import PostProcessPMBDB2PMBDBDL

from typing import Tuple, List, Iterator, Dict, Union
import pandas as pd
import numpy as np
import logging
import time


class YoloV5WithoutGTPostProcessor(YoloV5PostProcessorBase):
    """
    不包含 GT 数据的后处理接口类

    输入: OP 数据 + op2p_config + pmbdb2pmbdbdl_config, 其中后两者是 GT 阶段计算出的阈值
    输出: 预测的 box 信息
    """

    '''
    protected
    '''

    def _input_data_for_export(self) -> Iterator[Tuple[str, pd.DataFrame, List[str]]]:
        name_list = ["OP"]
        data_list = [self.op_data]
        columns_list = [self.DEFAULT_OP_COLUMNS]
        return zip(name_list, data_list, columns_list)

    '''
    protected
    '''

    def _verify_dataframe_columns(self) -> bool:
        # 确保数据已载入
        if self.op_data is None:
            return False
        # 列顺序可以不一致
        if set(self.op_data.columns) != set(self.DEFAULT_OP_COLUMNS):
            raise ValueError(f"OP 数据的列名不正确: {self.op_data.columns}")
        # 去除掉面积为 0 的预测框
        if "Area" in self.op_data.columns:
            self.op_data = self.op_data[self.op_data["Area"] > 0.0]
        return True

    '''
    public: 输入
    '''

    def load_from_numpy(self, op_data: np.array, op_cols=None):
        triple_line_logging("[load] 载入 numpy 格式的数据")
        self.op_data = pd.DataFrame(op_data, columns=op_cols or self.DEFAULT_OP_COLUMNS)
        return self

    def load_from_dataframe(self, op_data: pd.DataFrame):
        triple_line_logging("[load] 载入 dataframe 格式的数据")
        self.op_data = op_data
        return self

    def load_from_csv(self, op_path: str, header_mode: Union[str, None] = 'infer', op_cols=None):
        triple_line_logging("[load] 载入 csv 格式的数据")
        self.op_data = pd.read_csv(op_path, header=header_mode)
        # 默认 header
        if header_mode is None:
            self.op_data.columns = self.DEFAULT_OP_COLUMNS
        # 覆盖 header
        if op_cols:
            self.op_data.columns = op_cols
        return self

    def load_from_db(self, connect_options: dict, op_table: str = None):
        triple_line_logging("[load] 从数据库载入数据")
        reader = Reader(connect_options, silent=True)
        self.op_data = read_table_as_dataframe(reader, op_table or self.DEFAULT_OP_TABLE_NAME)
        return self

    '''
    public: 输出
    '''

    # 父类方法: export_to_csv

    # 父类方法: export_to_db

    def export_final_boxes_to_dataframe(self):
        if not self.process_list:
            raise ExportBeforeRunError("导出数据前，需要先执行后处理")
        p_name, p = self.process_list[-1]
        triple_line_logging("[export] 导出处理完成的 p-boxes 数据到 dataframe 变量")
        result = p.result.copy()
        if len(result) <= 0:
            result = pd.DataFrame(columns=PostProcessPMBDB2PMBDBDL.EXPORT_FIELD_NAMES)
        return result

    '''
    public: run
    '''

    def run(
            self, op2p_config: Dict[str, Threshold],
            pmbdb2pmbdbdl_config: Dict[str, Threshold],
    ):
        self._verify_dataframe_columns()
        time_start = time.time()
        logging.info("[start] 开始: 不包含 gt 数据的后处理流程")
        self.process_list = self._op_processes(op2p_config, pmbdb2pmbdbdl_config)
        logging.info(f"[end] 结束: 不包含 gt 数据的后处理流程，耗时 {time.time() - time_start:.4f}s")
        return self

    def run_with_parameter_files(self, from_folder: str):
        # 指定的参数文件夹下，应包含以下两个文件:
        # op2p_parameters.json
        # pmbdb2pmbdbdl_parameters.json
        return self.run(**self._load_parameter_dir_files(from_folder))
