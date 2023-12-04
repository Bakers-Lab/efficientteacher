from post_process_v2.mysql.connector import Connector
from post_process_v2.mysql.sql_builder import SQLBuilder
import logging

class Reader(Connector):
    YIELD_FETCH_SIZE = 500

    def __init__(self, connect_options: dict, silent=False):
        Connector.__init__(self, connect_options)
        self.silent = silent
        self.builder = SQLBuilder()

    def _calc_row_header(self, description, header=None):
        assert not header or len(header) == len(description)
        return header or [x[0] for x in description]

    '''
    public: read
    '''

    def readFrom(
            self, table_name, where_options=None, cols='*',
            limit=None, offset=None,
            override_header=None, auto_header=True, with_header=False,
    ):
        """
        从表格读取数据
        :param table_name: 表格名称
        :param where_options: 形如 ["id = 10"] 的条件限制的 list
        :param cols: 指定返回的列，string 或者 list，比如 "*" 或者 ["url"]
        :param limit: 返回行数限制
        :param offset: 偏移量
        :param override_header: 使用提供的表头覆盖表格的列
        :param auto_header: 自动将元组类型转换为字典类型
        :return: list
        """
        assert auto_header or not override_header
        description, result = self.fetch_all(
            sql=self.builder.read_sql(table_name, where_options, cols, limit, offset),
        )
        header = self._calc_row_header(description, override_header)
        if auto_header:
            result = [dict(zip(header, row)) for row in result]
        if with_header:
            result = (result, header)
        return result

    def distinctRead(
            self, table_name: str, filed_name: str, where_options=None
    ):
        description, result = self.fetch_all(
            sql=self.builder.read_sql(table_name, where_options, cols=f"DISTINCT `{filed_name}`"),
        )
        result = [row[0] for row in result]
        return result

    def fetch_all(self, sql):
        if not self.silent:
            logging.info(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.description, cursor.fetchall()
        self.db.commit()
        cursor.close()
        return result

    '''
    public: yield read
    '''

    def yield_read_from(
            self, table_name, where_options=None, cols='*',
            limit=None, offset=None,
            override_header=None, auto_header=True,
    ):
        assert auto_header or not override_header
        header = None
        for description, _row in self.yield_fetch_all(
                sql=self.builder.read_sql(table_name, where_options, cols, limit, offset),
        ):
            if auto_header:
                header = header or self._calc_row_header(description, override_header)
                _row = dict(zip(header, _row))
                yield _row
        return None

    def yield_distinct_read(
            self, table_name: str, filed_name: str, where_options=None
    ):
        for _row in self.fetch_all(
                sql=self.builder.read_sql(table_name, where_options, cols=f"DISTINCT `{filed_name}`"),
        ):
            yield _row[0]

    def yield_fetch_all(self, sql):
        if not self.silent:
            logging.info(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        while True:
            description, data = cursor.description, cursor.fetchmany(size=self.YIELD_FETCH_SIZE)
            if not data:
                break
            for _row in data:
                yield description, _row
        self.db.commit()
        cursor.close()
