# 调用方式

```shell
# 方式一: 使用同目录下的 config.json 文件
./post_process

# 方式二: 指定配置文件路径（相对/绝对路径均可，但不要使用 ~ 开头的路径形式） 
./post_process --config_path /var/www/data/config_v20230215.json
```

- 首次运行时，须确保该文件在系统中拥有执行权限。Linux 系统中可以通过 `chmod +x ./post_process` 命令为文件添加可执行权限。
- 当运行中抛出异常时，错误信息可在同路径下的 `post_process_errors.log` 日志文件中查看

# 配置文件说明（JSON 格式）

```json
{
    "label_id_name_dict": {
        "0": "U_FM",
        "1": "U_CM",
        "2": "U_PL",
        "3": "U_MC",
        "4": "U_PASS",
        "5": "U_OTHER"
    },
    "gt_csv_path": "/remote/avi/post_process_data_20230211/import/data/gt.txt",
    "op_csv_path": "/remote/avi/post_process_data_20230211/import/data/op.txt",
    "export_dir": "/remote/avi/post_process_data_20230211/cli-export",
    "pass_label_name": null
}
```

- `label_id_name_dict`: （必填字段）标签的 `ID-名称` 信息。标签的种类数量应与 GT 和 OP 数据一致。
- `gt_csv_path`: （必填字段）CSV 格式的 GT 数据文件路径。数据具体要求见下文。
- `op_csv_path`: （必填字段）CSV 格式的 OP 数据文件路径。数据具体要求见下文。
- `export_dir`: （必填字段）后处理结果的导出文件夹。后处理完成后，该文件夹中将包含:
    - 阈值文件：`op2p_parameters.json`, `pmbdb2pmbdbdl_parameters.json`
    - 标签级指标 `TR4Ratio.csv`
    - 图片级指标: `image_metrics.csv`
    - 图片级指标中对应的图片 ID 信息: `image_metric_ids.json`
    - 后处理中间结果: `TP.csv, TPMB.csv, TPMBDB.csv, TPMBDBDL.csv`
- `pass_label_name`: （可选字段）
    - 当此项留空（值为 `null`）时，将全部标签类别看作缺陷类。此时只生成标签级指标，不会生成图片级指标。
    - 当此项非空时，将指定的标签类别视为 Pass 类。此时将生成标签级指标，及图片级指标。

# 数据说明

- GT 和 OP 的数据文件，格式要求为不包含 header 信息的 CSV 格式
- GT 数据共 8 列，分别表示：`CenterX, CenterY, Length, Width, ImageID, LabelID, Area, BoxID`。形如：

```text
0.5105,0.6218,0.1396,0.1727,0,0,0.0241,0
0.2308,0.6954,0.1138,0.1566,0,0,0.0178,1
0.5086,0.6252,0.1568,0.2035,1,3,0.0319,0
0.2361,0.6794,0.1339,0.2128,1,3,0.0285,1
```

- OP 数据共 9 列，分别表示：`CenterX, CenterY, Length, Width, Confidence, ImageID, LableID, Area, BoxID`。形如：

```text
0.2330,0.6947,0.1159,0.1549,0.9552,0,0,0.0179,0
0.5118,0.6234,0.1424,0.1744,0.9506,0,0,0.0248,1
0.5125,0.6232,0.1463,0.1806,0.0036,0,1,0.0264,2
0.5121,0.6231,0.1430,0.1735,0.0035,0,3,0.0248,3
```