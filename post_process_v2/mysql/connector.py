import pymysql
import logging


class Connector:
    CONNECT_OPT_KEYS = ("host", "port", "user", "password", "db")

    def __init__(self, connect_options: dict):
        self.db = self.__init_connect(connect_options)

    def __init_connect(self, options: dict):
        return pymysql.connect(**{x: options[x] for x in self.CONNECT_OPT_KEYS})

    def transaction(self, sql_list, print_sql=True, commit_callback=None):
        result = True
        if isinstance(sql_list, str):
            sql_list = [sql_list]
        try:
            cursor = self.db.cursor()
            for sql in sql_list:
                if print_sql:
                    logging.info(sql)
                cursor.execute(sql)
            self.db.commit()
            if commit_callback:
                result = commit_callback(cursor)
            cursor.close()
        except:
            logging.info("事务执行失败，回滚")
            self.db.rollback()
            result = False
        return result

    def close(self):
        self.db.close()
