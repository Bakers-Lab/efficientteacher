from post_process_v2.base.helpers import write_json_to_disk, method_run_once_limit, dataframe_yield_as_dict_rows
from post_process_v2.base.exceptions import ExportBeforeRunError
from post_process_v2.procedures.init_parameters import Threshold

from typing import Dict, Tuple
import pandas as pd
import os.path
import logging


class PostProcessAnalysisData:
    DEFAULT_MIN_CONFIDENCE = 1.0
    MIN_CONFIDENCE_OFFSET = -0.0001

    def __init__(self, id_name_dict: Dict[int, str], op_data: pd.DataFrame, pmbdb_data: pd.DataFrame):
        self.id_name_dict = id_name_dict
        self.op_data = self._parse_raw_data(op_data)
        self.pmbdb_data = self._parse_raw_data(pmbdb_data)
        # values
        self._op_thresholds, self._pmbdb_thresholds = None, None

    @staticmethod
    def export_one_threshold(data: Dict[str, Threshold], dest_path: str):
        write_json_to_disk(
            dest_path=dest_path,
            data={x: y.to_dict() for x, y in data.items()},
        )
        logging.info(f"写阈值配置文件到 {dest_path}")

    '''
    protected
    '''

    def _parse_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[data["p_label_id"] == data["gt_label_id"]]

    def _build_op_thresholds(self) -> Dict[str, Threshold]:
        threshold_dict = {
            _id: Threshold(_id, _name, min_confidence=self.DEFAULT_MIN_CONFIDENCE)
            for _id, _name in self.id_name_dict.items()
        }
        # 按照 label-id 分组
        for _row in dataframe_yield_as_dict_rows(self.op_data):
            key = int(_row["gt_label_id"])
            threshold_dict[key].min_confidence = min(
                threshold_dict[key].min_confidence,
                _row["confidence"] + self.MIN_CONFIDENCE_OFFSET,
            )
        # 构造返回值
        result = {_name: threshold_dict[_id] for _id, _name in self.id_name_dict.items()}
        return result

    def _build_pmbdb_thresholds(self) -> Dict[str, Threshold]:
        threshold_dict = {
            _id: Threshold(_id, _name, min_confidence=dict())
            for _id, _name in self.id_name_dict.items()
        }
        # 按照 (label-id, label-index) 分组
        for _row in dataframe_yield_as_dict_rows(self.pmbdb_data):
            min_confidence = threshold_dict[int(_row["gt_label_id"])].min_confidence
            min_confidence[int(_row["p_label_index"])] = min(
                _row["confidence"] + self.MIN_CONFIDENCE_OFFSET,
                min_confidence.get(int(_row["p_label_index"]), self.DEFAULT_MIN_CONFIDENCE),
            )
        # 构造返回值
        result = {_name: threshold_dict[_id] for _id, _name in self.id_name_dict.items()}
        return result

    '''
    public
    '''

    def export_to_json(self, dest_dir: str):
        if not self._op_thresholds or not self._pmbdb_thresholds:
            raise ExportBeforeRunError("导出阈值数据前，须先执行数据分析")
        # 构造数据
        data_list = (self._op_thresholds, self._pmbdb_thresholds)
        name_list = ("op2p_parameters", "pmbdb2pmbdbdl_parameters")
        # 写到磁盘
        for _data, _name in zip(data_list, name_list):
            self.export_one_threshold(_data, os.path.join(dest_dir, f"{_name}.json"))

    @method_run_once_limit
    def run(self):
        self._op_thresholds = self._build_op_thresholds()
        self._pmbdb_thresholds = self._build_pmbdb_thresholds()
        return self

    @property
    def result(self) -> Tuple[Dict[str, Threshold], Dict[str, Threshold]]:
        return self._op_thresholds, self._pmbdb_thresholds
