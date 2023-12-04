from post_process_v2.base.helpers import create_dir_if_not_exists, get_filename_without_ext, \
    write_dataframe_to_db
from post_process_v2.base.exceptions import ExportBeforeRunError
from post_process_v2.mysql.writer import Writer

from abc import ABC, abstractmethod
import pandas as pd
import logging
import os


class BaseProcedure(ABC):
    EXPORT_FIELD_NAMES = []

    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self._result = None

    '''
    public
    '''

    def export_to_csv(self, csv_path: str, with_header=True):
        # 数据校验
        if not self.EXPORT_FIELD_NAMES:
            raise ValueError("导出的列名未指定")
        if self._result is None:
            raise ExportBeforeRunError("导出数据前，须先执行处理")
        if len(self._result) <= 0:
            logging.warning(f"处理阶段 {get_filename_without_ext(csv_path)} 的数据为空，跳过数据导出到 CSV")
            return None
        create_dir_if_not_exists(os.path.dirname(csv_path))
        # 构造数据
        data = self._build_export_data()[self.EXPORT_FIELD_NAMES]
        # 写到磁盘
        data.to_csv(csv_path, header=with_header, index=False)
        logging.info(f"写 {get_filename_without_ext(csv_path)} 数据到 {csv_path}")

    def export_to_db(self, writer: Writer, name: str, table_prefix: str):
        if not self.EXPORT_FIELD_NAMES:
            raise ValueError("导出的列名未指定")
        if self._result is None:
            raise ExportBeforeRunError("导出数据前，须先执行处理")
        write_dataframe_to_db(
            writer, name, table_prefix,
            data=self._build_export_data(), filed_names=self.EXPORT_FIELD_NAMES,
        )

    @property
    def result(self) -> pd.DataFrame:
        if self._result is None:
            raise ExportBeforeRunError("获取处理结果前，须先执行处理")
        return self._result

    '''
    protected
    '''

    def _build_export_data(self) -> pd.DataFrame:
        return self._result

    '''
    必须实现的方法
    '''

    @abstractmethod
    def run(self):
        raise NotImplementedError()
