from post_process_v2.base.helpers import write_dataframe_to_db, triple_line_logging, \
    load_json_from_file, reverse_dict
from post_process_v2.base.exceptions import ExportBeforeRunError
from post_process_v2.mysql.writer import Writer

from post_process_v2.procedures.init_parameters import Threshold
from post_process_v2.procedures.op_to_p import PostProcessOP2P
from post_process_v2.procedures.p_to_pmb import PostProcessP2PMB
from post_process_v2.procedures.pmb_to_pmbdb import PostProcessPMB2PMBDB
from post_process_v2.procedures.pmbdb_to_pmbdbdl import PostProcessPMBDB2PMBDBDL

from typing import Tuple, List, Iterator, Dict
from abc import ABC, abstractmethod
import pandas as pd
import os.path
import logging


class YoloV5PostProcessorBase(ABC):
    DEFAULT_OP_TABLE_NAME = "TOP"
    DEFAULT_GT_TABLE_NAME = "TGT"

    DEFAULT_GT_COLUMNS = ["CenterX", "CenterY", "Length", "Width", "ImageID", "LableID", "Area", "BoxID"]
    DEFAULT_OP_COLUMNS = ["CenterX", "CenterY", "Length", "Width", "Confidence", "ImageID", "LableID", "Area", "BoxID"]

    PARAMETER_KEYS = ("op2p", "pmbdb2pmbdbdl")

    def __init__(
            self, label_id_name_dict: Dict[int, str], p2pmb_thresholds: Dict[str, float],
            pass_label_name: str = None, pass_override_min: float = 0.75,
    ):
        if pass_label_name and pass_label_name not in list(label_id_name_dict.values()):
            raise ValueError(f"无效的 pass_label_name={pass_label_name}")
        if pass_label_name and not (0.0 < pass_override_min < 1.0):
            logging.warning("pass_override_min 的值不属于 (0.0, 1.0) 开区间范围，将不会生效")
        # params
        self.label_id_name_dict = label_id_name_dict
        self.p2pmb_thresholds = p2pmb_thresholds
        self.pass_label_name = pass_label_name
        self.pass_override_min = pass_override_min
        # values
        self.pass_label_id = self.__calc_pass_label_id()
        self.op_data = None
        self.process_list = []

    '''
    protected: 核心处理过程
    '''

    def _op_processes(self, op2p_config: Dict[str, Threshold], pmbdb2pmbdbdl_config: Dict[str, Threshold]):
        # op -> p
        op2p = PostProcessOP2P(self.op_data, op2p_config).run()
        logging.info("处理流程结束: op -> p")
        # p -> pmb
        p2pmb = PostProcessP2PMB(op2p.result, **self.p2pmb_thresholds).run()
        logging.info("处理流程结束: p -> pmb")
        # pmb -> pmbdb
        pmb2pmbdb = PostProcessPMB2PMBDB(p2pmb.result).run()
        logging.info("处理流程结束: pmb -> pmbdb")
        # pmbdb -> pmbdbdl
        pmbdb2pmbdbdl = PostProcessPMBDB2PMBDBDL(
            pmb2pmbdb.result, pmbdb2pmbdbdl_config, self.pass_label_id, self.pass_override_min,
        ).run()
        logging.info("处理流程结束: pmbdb -> pmbdbdl")
        # 记录中间流程
        return [("P", op2p), ("PMB", p2pmb), ("PMBDB", pmb2pmbdb), ("PMBDBDL", pmbdb2pmbdbdl)]

    '''
    private
    '''

    def __calc_pass_label_id(self):
        # Pass 标签信息
        result = None
        if self.pass_label_name:
            result = reverse_dict(self.label_id_name_dict)[self.pass_label_name]
        return result

    '''
    protected: 导入导出相关
    '''

    def _load_parameter_dir_files(self, from_folder: str) -> Dict[str, Dict[str, Threshold]]:
        logging.info(f"使用 {from_folder} 文件夹下的阈值配置文件")
        result = dict()
        for _key in self.PARAMETER_KEYS:
            _path = os.path.join(from_folder, f"{_key}_parameters.json")
            result[f"{_key}_config"] = {str(x): Threshold(**y) for x, y in load_json_from_file(_path).items()}
        return result

    def _export_input_data_to_db(self, writer: Writer, table_prefix: str):
        for _name, _data, _columns in self._input_data_for_export():
            write_dataframe_to_db(writer, _name, table_prefix, _data, _columns)

    def _export_input_data_to_csv(self, dest_dir: str, with_header):
        for _name, _data, _columns in self._input_data_for_export():
            _path = os.path.join(dest_dir, f"T{_name}.csv")
            if not len(_data):
                logging.warning(f"{_name} 数据为空，跳过导出")
                continue
            _data.to_csv(_path, header=with_header, index=False)
            logging.info(f"写 {_name} 数据到 {_path}")

    '''
    public: 导出数据
    '''

    def export_to_csv(self, dest_dir: str, with_header=True, with_input_data=False):
        if not self.process_list:
            raise ExportBeforeRunError("导出数据前，需要先执行后处理")
        triple_line_logging("[export] 导出数据到 csv 文件中")
        for name, processor in self.process_list:
            _path = os.path.join(dest_dir, f"T{name}.csv")
            processor.export_to_csv(_path, with_header)
        # OP|GT 数据
        if with_input_data:
            self._export_input_data_to_csv(dest_dir, with_header)

    def export_to_db(self, connect_options: dict, table_prefix: str = "T", with_input_data=False):
        if not self.process_list:
            raise ExportBeforeRunError("导出数据前，需要先执行后处理")
        if len(table_prefix) > 50:
            raise ValueError("table_prefix 的长度超出限制")
        triple_line_logging("[export] 导出数据到数据库")
        writer = Writer(connect_options, silent=True)
        for _name, p in self.process_list:
            p.export_to_db(writer, _name, table_prefix)
        # OP|GT 数据
        if with_input_data:
            self._export_input_data_to_db(writer, table_prefix)

    '''
    必须实现的方法
    '''

    @abstractmethod
    def _input_data_for_export(self) -> Iterator[Tuple[str, pd.DataFrame, List[str]]]:
        # (name, data, columns)
        raise NotImplementedError()
