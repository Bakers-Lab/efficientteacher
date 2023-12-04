# 数据库连接
DEFAULT_CONNECT_OPTIONS = {
    "host": "<host>",
    "port": 3306,
    "password": "<password>",
    "user": "root",
    "db": "<db>",
}

# YoloV5PostProcessor 类的参数
label_id_name_dict = {
    0: "DZ_CM",
    1: "DZ_FM",
    2: "DZ_MC",
    3: "DZ_PASS",
}
p2pmb_thresholds = {
    "area_threshold": 0.05,
    "length_threshold": 0.25,
    "iou_threshold": 0.9,
}
