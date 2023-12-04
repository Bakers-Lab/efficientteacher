from datetime import date, datetime
import json


class SQLBuilder:
    def __wrap_string(self, string):
        string = string.replace('"', '\\"').replace("'", "\\'")
        while string.endswith("\\"):
            string = string[:-1]
        return f"'{string}'"

    def _wrap_value(self, v):
        if isinstance(v, str):
            result = self.__wrap_string(v)
        elif v is None:
            result = 'null'
        elif isinstance(v, dict) or isinstance(v, list) or isinstance(v, tuple):
            result = json.dumps(v, ensure_ascii=False)
            result = self.__wrap_string(result)
        elif isinstance(v, datetime) or isinstance(v, date):
            result = self.__wrap_string(str(v))
        elif isinstance(v, bool):
            result = "1" if v else "0"
        else:
            result = str(v)
        return result

    def _set_key_value_str(self, key_values: dict):
        return ", ".join([
            f"`{key}` = {self._wrap_value(value)}"
            for key, value in key_values.items()
        ])

    def _create_cols(self, columns):
        result = columns
        if not isinstance(columns, str):
            result = ', '.join(["`{}`".format(col) for col in columns])
        return result

    '''
    public 方法: sql
    '''

    def read_sql(self, table_name, where_options=None, cols='*', limit=None, offset=None):
        result = f"SELECT {self._create_cols(cols)} FROM `{table_name}`"
        result += " WHERE " + ' '.join(where_options) if where_options and len(where_options) > 0 else ''
        result += f" LIMIT {limit}" if limit else ''
        result += f" OFFSET {offset}" if offset else ''
        return result

    def write_sql(self, table_name, row):
        return "INSERT INTO {}({}) VALUES ({})".format(
            table_name,
            ','.join([f"`{x}`" for x in row.keys()]),
            ','.join([self._wrap_value(x) for x in row.values()])
        )

    def update_sql(self, table_name: str, values: dict, *where_options):
        return f"UPDATE {table_name} SET {self._set_key_value_str(values)} WHERE {' '.join(where_options)}"

    def update_by_key_sql(self, table_name: str, values: dict, p_key: str, updated_at=None):
        where_options = [f"`{p_key}` = {values[p_key]!r}"]
        if updated_at:
            values["updated_at"] = updated_at
            where_options.extend(["AND", f"(`updated_at` IS NULL OR `updated_at` > {updated_at!r})"])
        return self.update_sql(table_name, values, *where_options)

    def delete_sql(self, table_name: str, *where_options):
        return f"DELETE FROM {table_name} WHERE {' '.join(where_options)}"

    def multi_update_sql(
            self, table_name, select_key: str, id_values: dict,
            single_field=None, where_options=None,
    ):
        """
        [1] 指定 single_field
        id_values: {<key>: <value>} == {select_key: single_field 字段的值 }, SQL 如下:
        UPDATE `table_name` SET `single_field` = <value> WHERE `select_key` = <key>;
        --------------------
        [2] single_field is None
        id_values: {<key>: <value>} == {select_key: 多个字段的键值对 }, SQL 如下:
        UPDATE `table_name` SET
            `<value.k1>` = <value.v1> AND
            `<value.k2>` = <value.v2> AND ...
        WHERE `select_key` = <key>;
        """
        where_options = self.wrap_where_options(where_options, append_and=True)
        # 构建 SQL
        result = []
        for _id, value in id_values.items():
            if not (isinstance(value, dict) or single_field):
                raise Exception(f"当 `id_values.value`: {value} 不是 `dict` 类型时，必须指定 `single_field` 参数")
            # 包裹单字段数据
            if single_field:
                value = {single_field: value}
            # 构建 SQL 语句
            result.append(
                self.update_sql(
                    table_name, value,
                    *where_options,
                    f"`{select_key}` = {_id!r}",
                )
            )
        return result

    '''
    static 方法: 辅助方法
    '''

    @staticmethod
    def wrap_where_options(where_options, append_and=False):
        if not where_options:
            return []
        result = list(where_options)
        result.insert(0, " ( ")
        result.append(" ) ")
        if append_and:
            result.append(" AND ")
        return result
