from post_process_v2.mysql.connector import Connector
from post_process_v2.mysql.sql_builder import SQLBuilder

import logging

class Writer(Connector):
    def __init__(self, connect_options: dict, silent=False):
        Connector.__init__(self, connect_options)
        self.silent = silent
        self.sql_builder = SQLBuilder()

    def writeTo(self, table_name, data, with_last_id=False, updated_at=None):
        def callback(cursor):
            if not with_last_id:
                return True
            cursor.execute("select LAST_INSERT_ID();")
            return cursor.fetchone()[0]

        if updated_at:
            data["updated_at"] = updated_at
        return self.transaction(
            [self.sql_builder.write_sql(table_name, data)],
            commit_callback=callback,
            print_sql=(not self.silent),
        )

    def writeManyTo(
            self, table_name, data,
            retry_each=False, row_callback=None, updated_at=None,
    ):
        """
        一次事务，插入多条数据
        :param table_name: 表名
        :param data: 多行数据
        :param retry_each: 如果一次写入多行失败，则尝试逐行写入
        :param row_callback: 写入成功的行的回调
        :return: int
        """
        if updated_at:
            for _row in data:
                _row["updated_at"] = updated_at
        # 合并写
        result = self.transaction(
            [self.sql_builder.write_sql(table_name, row) for row in data],
            print_sql=(not self.silent),
        )
        result = len(data) if result else 0
        # 合并写成功
        if result and row_callback:
            for row in data:
                row_callback(row)
        # 合并写失败，尝试逐行写入
        if not result and retry_each and data:
            logging.warning("批量写入失败，开始尝试逐条写入")
            for row in data:
                if not self.writeTo(table_name, row):
                    continue
                result += 1
                if row_callback:
                    row_callback(row)
            logging.info(f"本轮逐条写入合计成功 {result} 条")
        return result

    def update(self, table_name, values: dict, where_options: list):
        return self.transaction(
            sql_list=[self.sql_builder.update_sql(table_name, values, *where_options)],
            print_sql=(not self.silent),
        )

    def update_by_key(self, table_name, row_data: dict, p_key="id", updated_at=None):
        return self.transaction(
            sql_list=[self.sql_builder.update_by_key_sql(table_name, row_data, p_key, updated_at)],
            print_sql=(not self.silent),
        )

    def updateManyByPrimaryKey(
            self, table_name, id_values: dict,
            single_field=None, select_key="id",
            where_options=None,
    ):
        sql_list = self.sql_builder.multi_update_sql(
            table_name, select_key, id_values,
            single_field=single_field,
            where_options=where_options,
        )
        return self.transaction(sql_list, print_sql=(not self.silent))

    def delete(self, table_name, where_options: list):
        return self.transaction(
            self.sql_builder.delete_sql(table_name, *where_options),
            print_sql=(not self.silent),
        )
