from post_process_v2.mysql.reader import Reader
from post_process_v2.mysql.writer import Writer
from post_process_v2.base.exceptions import RunTwiceError

from typing import List, Dict, Any, Union, Callable
import pandas as pd
import logging
import json
import os

'''
路径相关
'''


def yield_walk_files(root: str):
    for parent, dir_names, file_names in os.walk(root):
        for _name in file_names:
            yield os.path.join(parent, _name)
    return None


def create_dir_if_not_exists(folder_path: str) -> str:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def get_filename_without_ext(path) -> str:
    return os.path.basename(os.path.splitext(path)[0]).strip()


def build_sql_template_path(name: str) -> str:
    result = None
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sql_templates")
    for _path in yield_walk_files(root):
        if get_filename_without_ext(_path).lower() == name.lower():
            result = _path
            break
    return result


'''
数据结构相关
'''


def dataframe_yield_as_dict_rows(data: pd.DataFrame, filter_func: Callable[[Dict], bool] = None):
    for _row in data.itertuples(index=False):
        _dict_row = dict(zip(data.columns, _row))
        if filter_func and not filter_func(_dict_row):
            continue
        yield _dict_row
    return None


def dataframe_group_by(
        data: pd.DataFrame, by: Union[str, List[str]],
        filter_func: Callable[[Dict], bool] = None,
) -> Dict[Any, List[Dict]]:
    result = dict()
    for _row in dataframe_yield_as_dict_rows(data, filter_func):
        _key = _row[by] if isinstance(by, str) else tuple([_row[x] for x in by])
        result[_key] = result.get(_key, [])
        result[_key].append(_row)
    return result


def read_table_as_dataframe(reader: Reader, table_name: str) -> pd.DataFrame:
    data, header = reader.readFrom(table_name, with_header=True)
    return pd.DataFrame(data, columns=header)


def group_dict_list_by(data: List[Dict], key: str) -> Dict[Any, List]:
    result = dict()
    for _row in data:
        result[_row[key]] = result.get(_row[key], [])
        result[_row[key]].append(_row)
    return result


def reverse_dict(data: Dict):
    return {y: x for x, y in data.items()}


'''
读写相关
'''


def load_json_from_file(path: str) -> dict:
    with open(path, "rt", encoding="utf8") as f:
        result = json.load(f)
    return result


def write_json_to_disk(data: dict, dest_path: str) -> str:
    create_dir_if_not_exists(os.path.dirname(dest_path))
    with open(dest_path, "wt", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return dest_path


def write_dataframe_to_db(
        writer: Writer, name: str, table_prefix: str,
        data: pd.DataFrame, filed_names: list,
) -> int:
    if not filed_names:
        raise ValueError("要写入的字段名列表不能为空")
    table_name = table_prefix + name
    # 建表
    sql_tpl_path = build_sql_template_path(name)
    with open(sql_tpl_path, "rt", encoding="utf8") as f:
        create_table_sql = f.read().replace("<table_name>", table_name)
    writer.transaction([
        f"DROP TABLE IF EXISTS `{table_name}`;", create_table_sql,
    ], print_sql=False)
    # 构建行数据
    rows = []
    for _row in dataframe_yield_as_dict_rows(data):
        rows.append({x: _row[x] for x in filed_names})
    status = writer.writeManyTo(table_name, rows)
    if not status:
        logging.error(f"写入到 {table_name} 表失败")
    else:
        logging.info(f"共计成功写入 {len(rows)} 条数据到数据库，table={table_name}")
    return status


'''
辅助函数
'''


def triple_line_logging(msg, *args, **kwargs):
    logging.info('-' * 88)
    logging.info(msg, *args, **kwargs)
    logging.info('-' * 88)


def method_run_once_limit(func):
    # 装饰器: 对于 func 所属对象，给定函数 func 最多调用一次
    times = dict()

    def wrapper(self, *args, **kwargs):
        times[self] = times.get(self, 0)
        if times[self] > 0:
            raise RunTwiceError("此函数只允许被调用一次")
        result = func(self, *args, **kwargs)
        times[self] += 1
        return result

    return wrapper
