# TODO

- [] 拼写错误: `LableID`, `LableIndex`, `LableName`
- [x] ~~使用 abstract class 作为统一的 procedure class 规约~~
- [x] ~~每个阶段 export 的 csv 文件限制为只有一个~~
- [x] ~~导出为变量；导出时版本的区分~~
- [x] ~~没有 GT 数据的处理流程~~
- [x] ~~在更大数据集上进行耗时测试~~

# 使用说明

### 简介

- 基本使用流程: `载入数据 -> 执行后处理 -> 导出处理结果`
- 支持通过 `numpy.array`, `pandas.DataFrame`, `CSV 文件`, `数据库连接` 四种方式载入数据
- 支持通过 `CSV 文件`, `数据库连接` 两种持久化方式，导出中间处理结果及相关 `metrics`

### `YoloV5WithGTPostProcessor` 类

1. 位于 `post_process_v2.api.with_gt.py` 文件
2. 使用 `op` 和 `gt` 数据作为输入
3. 支持以持久化方式导出的数据包括:
    - 各处理阶段的中间结果及 metrics: `P`, `PMB`, `PMBDB`, `PMBDBDL` + `R4Ratio`。通过调用 `export_to_csv` 或 `export_to_db` 方法导出
    - `op2p` 和 `pmbdb2pmbdbdl` 阶段的阈值。通过设置 `run` 方法的 `export_parameters_dir` 参数，指定要保存到的文件夹。这两个阈值会被保存为 `JSON`
      格式的文件，分别为 `op2p_parameters.json` 和 `pmbdb2pmbdbdl_parameters.json`
4. 支持以内存变量方式导出的数据包括:
    - 各处理阶段的 `metrics`。调用 `export_ratio_to_dataframe` 方法，返回 `pandas.DataFrame` 类型的数据
5. 支持 `自动计算阈值` 或 `指定阈值` 两种调用方式
    - `自动计算阈值` 方式: 共两轮的后处理过程。调用 `run` 方法，会输出计算生成的阈值到指定文件夹
    - `指定阈值` 方式: 共一轮的后处理过程。调用 `run_with_parameters` 或 `run_with_parameters_from_dir` 方法
    - 上述两种方式中的 `阈值`，具体指 `op2p` 和 `pmbdb2pmbdbdl` 两个阶段的阈值参数，具体说明见下文。

### `YoloV5WithoutGTPostProcessor` 类

1. 位于 `post_process_v2.api.without_gt.py` 文件
2. 使用 `op` 数据作为输入
3. 支持以持久化方式导出的数据包括:
    - 各处理阶段的中间结果: `P`, `PMB`, `PMBDB`, `PMBDBDL`。通过调用 `export_to_csv` 或 `export_to_db` 方法导出
4. 支持以内存变量方式导出的数据包括:
    - 处理完成后（`PMBDBDL` 阶段后），最终的检测框数据。调用 `export_final_boxes_to_dataframe` 方法，返回 `pandas.DataFrame` 类型的数据
5. 允许各分类标签中存在 0 或 1 种 Pass 标签。当存在 Pass 标签时，通过 `YoloV5WithGTPostProcessor` 类构造函数中的 `pass_label_name` 参数指定。详细的指标计算规则见下文。

### 配置相关说明

1. 数据库连接: 以 `dict` 类型变量作为参数传入，示例参见 `env_example.py` 文件。数据库配置并无存储位置的限制，在使用时作为参数变量传入即可。
2. `YoloV5WithGTPostProcessor` 和 `YoloV5WithoutGTPostProcessor` 需要传入 `label_id_name_dict` 和 `p2pmb_thresholds`
   变量作为初始化参数，具体格式见 `env_example.py` 文件

### `op2p` 和 `pmbdb2pmbdbdl` 阈值

1. 对于包含 `gt` 数据的后处理流程的 `自动计算阈值` 方式，这两种阈值不需要手动设置。
1. 对于包含 `gt` 数据的后处理流程的 `指定阈值` 方式，需要明确指定阈值文件所在文件夹，或者将阈值变量作为参数传入。
2. 对于不包含 `gt` 数据的后处理流程，需要明确指定阈值文件所在文件夹，或者将阈值变量作为参数传入。
3. 对于单个类别的阈值，它的数据结构被封装为 `Threshold` 类。
    - 具体定义见 `post_process_v2/procedures/init_parameters.py` 文件。示例见其中的 `parameters_example` 函数
    - `Threshold.min_confidence` 属性表示最小阈值，即: 在筛选时，只有 `Confidence` **严格大于** 它才能通过。
    - `op2p` 阶段需要对每个 `Label` 设置一个阈值，此时 `min_confidence` 为 `float` 类型
    - `pmbdb2pmbdbdl` 阶段可能需要对每个 `(Label, LabelIndex)` 设置一个阈值，此时 `min_confidence` 为 `Dict[int, float]` 类型
4. 导出的阈值配置文件，包括多个类别的阈值，示例如下（JSON 格式）

```json
{
    "DZ_CM": {
        "label_id": 0,
        "label_name": "DZ_CM",
        "min_confidence": {
            "1": 0.027800000000000002,
            "3": 0.0022,
            "2": 0.008700000000000001
        }
    },
    "DZ_FM": {
        "label_id": 1,
        "label_name": "DZ_FM",
        "min_confidence": 1.0
    }
}
```

# Code Example

### 载入数据

```python
from post_process_v2.api.with_gt import YoloV5WithGTPostProcessor
import os.path

API = YoloV5WithGTPostProcessor(label_id_name_dict=dict(), p2pmb_thresholds=dict())

# 通过 CSV 文件 载入
csv_src_root = "/var/data/yolov5/v20221201"
API.load_from_csv(
    os.path.join(csv_src_root, "op.csv"),
    os.path.join(csv_src_root, "gt.csv"),
    header_mode=None,
)

# 通过 数据库 载入
DEFAULT_CONNECT_OPTIONS = {
    "host": "<host>",
    "port": 3306,
    "password": "<password>",
    "user": "root",
    "db": "<db>",
}
API.load_from_db(DEFAULT_CONNECT_OPTIONS, op_table="top", gt_table="tgt")

```

### 执行后处理

1. 包含 `gt` 数据的后处理流程

- `自动计算阈值` 方式: 调用 `YoloV5WithGTPostProcessor.run` 函数。其中 `export_parameters_dir` 参数，表示 `op2p` 和 `pmbdb2pmbdbdl`
  两个阈值配置文件要保存到的文件夹
- `指定阈值` 方式: 调用 `YoloV5WithGTPostProcessor.run_with_parameters` 函数
  或 `YoloV5WithGTPostProcessor.run_with_parameters_from_dir` 函数。

2. 不包含 `gt` 数据的后处理流程

- 方式一: 调用 `YoloV5WithoutGTPostProcessor.run_with_parameter_files` 函数。`from_folder` 参数，表示两个阈值配置文件所在的文件夹路径
- 方式二: 调用 `YoloV5WithoutGTPostProcessor.run` 函数。两个阈值以 `Dict[str, Threshold]` 类型的参数传入

```python
from post_process_v2.api.with_gt import YoloV5WithGTPostProcessor

API = YoloV5WithGTPostProcessor(label_id_name_dict=dict(), p2pmb_thresholds=dict())

API.run(export_parameters_dir="/var/data/yolov5/v20221201-parameters")
```

### 导出数据到持久化存储介质

- **从命名上区分不同数据源**: 导出到 `CSV` 文件时，可以指定目标文件夹；导出到数据库时，可以指定数据库表的前缀。
- 导出的数据内容包括: `P`, `PMB`, `PMBDB`, `PMBDBDL` 这些中间处理结果和性能指标 `R4Ratio`
- 设置参数 `with_input_data=True` 时，导出内容将也会包括 `OP` 和 `GT` 数据

```python
from post_process_v2.api.with_gt import YoloV5WithGTPostProcessor

API = YoloV5WithGTPostProcessor(label_id_name_dict=dict(), p2pmb_thresholds=dict())

# ...

# 导出为 CSV 文件
csv_dest_root = "/var/data/yolov5/v20221201-post_process"
API.export_to_csv(csv_dest_root, with_header=False)

# 导出到数据库
DEFAULT_CONNECT_OPTIONS = {
    # ...
}
API.export_to_db(DEFAULT_CONNECT_OPTIONS, table_prefix="v20221201_", with_input_data=False)
```

# 原 MySQL Procedures 逻辑简述

### op -> p

- 移除 `confidence` 过低的 box
- 每个分类都有一个阈值

### p -> pmb

- 根据 area, length 合并框（新框过大则不合并）
- 根据 IoU 合并框
- 新框可能在多个分类上有 `置信度`。置信度 = 合并过的框中在该类上的 max(confidence)

### pmb -> pmbdb

- 为全部记录添加全局 ID（从 0 起始）
- 每个 box 可以有多行记录。根据这些行在各类标签上的置信度，逆序排序后，排名记为 LabelIndex（从 1 起始）

### pmbdb -> pmbdbdl

- 根据阈值，移除 `confidence` 过低的 box
- 每个分类都有一个阈值，或者为每个 (Label, LabelIndex) 设置阈值

### match_with_gt

- 计算与 GT 数据中存在重合的框
- 备选的框包括来自 `op, p, pmb, pmbdb, pmbdbdl` 各阶段的数据

### report4ratio

1. 子过程 StatisticsGT_Stage
    - 根据中间表 `TGT_{p}_Mapping` 生成拼接数据表 `SGT_{p}`，表示 gt 与 box 的一对多关系
    - 根据 `SGT_{p}`，生成 `SGT_{p}_group`，聚合表示一对多关系为一行

2. 子过程 Operator4Ratio
   指定要处理的 gt-label 集合后:
    1. 基于 `SGT_{p}_Group` 生成
        - `SGT_{p}_Missing`: gt-box 无对应 p-label/p-box
        - `SGT_{p}_Wrong`: gt-label 不在 p-label 集合中
        - `SGT_{p}_Match`: gt-label 在 p-label 集合中
    2. 基于 `TGT_{p}_Mapping` 生成
        - `SGT_{p}_Overkill`: 无对应 gt-box 的 p-box，即存在于 `T{p}` 中但是不存在于 `TGT_{p}_Mapping`
    3. 统计数据（用于输出）:
        - @1: gt-box 总数
        - @2: `T{p}` 中 p-box 总数（p-box 可以根据多种标签计算多次）
        - @3: `T{p}` 中 p-box 总数（每个 p-box 最多计一次）
        - @4: `SGT_{p}_Match` 中记录数量
        - @5: `SGT_{p}_Missing` 中记录数量
        - @6: `SGT_{p}_Wrong` 中记录数量
        - @7: `SGT_{p}_Overkill` 中记录数量
    4. 统计数据（用于修正）:
        - @8: `SGT_{p}_Missing` 中 label 属于 `PassFields` 的记录数量
        - @9: `SGT_{p}_Wrong` 中 label 属于 `PassFields` 的记录数量
        - @10: `SGT_{p}_Overkill` 中 label 属于 `PassFields` 的记录数量
    5. 修正规则
        - @5 -= @8, @6 -= @9, @7 -= @10
        - @4 += @8 + @9

3. 整体处理方法
    - 统计 `op, p, pmb, pmbdb, pmbdbdl` 多个阶段的 metric 信息
    - 对于每个阶段，分别统计以下条件下的 metric 信息:
        - `全部类别`
        - `全部类别（不含 other 类型）`
        - `单个类别`

### analysis data

1. 计算 op2p 阈值
    1. 对 `SGT_OP` 中预测正确的框（即 gt-label == p-label），按照 label-id 分组
    2. 每组设置 `最小置信度阈值` = `min(Confidence) - 0.0001`
    3. 如果组内无数据，则设置该组 `最小置信度阈值` = `1.00`

2. 计算 pdbdb2pmbdbdl 阈值
    1. 对 `SGT_PMBDB` 中预测正确的框，按照 (label-id, label-index) 分组
    2. 剩余步骤同 op2p 阈值计算

# 标签级指标计算方法（全部为缺陷标签）

### 符号规定

1. `gt-box` 表示 GT 数据中的目标检测框，`p-box` 表示预测框。
2. 框与框 `有交集`，表示两个框在坐标上的重叠面积大于 0。
3. `gt-label` 表示 gt 框的标签分类，`p-label` 表示预测框的标签
4. `{p-label}` 表示单个 gt-box 对应的多个 p-box 的标签集合。
5. `defect_i` 表示第 i 种 defect 标签。`{defect_i}` 表示全部 defect 标签的集合。

### 基本概念

1. `一致（match）`: (gt-box 与 p-box 有交集) AND (gt-label ∈ {p-label})
2. `漏检（missing）`: (gt-box 无 p-box 与之有交集)
3. `错检（wrong）`: (gt-box 与 p-box 有交集) AND (gt-label ∉ {p-label})
4. `过杀（过检）（over-kill）`: (无 gt-box 与 p-box 有交集)

### 说明

1. `一致率`、`漏检率`、`错检率` 的分母为 gt-box 的总数，且 `一致率 + 漏检率 + 错检率 = 100%`
2. `过杀率` 的分母为 p-box 的总数。坐标相同但标签不同的 p-box 被视为为多个。

# 标签级指标计算方法（带 Pass 标签）

### defect 标签的指标修正规则

1. Pass 标签错检为 defect 标签
    - 即：(gt-box 与 p-box 有交集) AND (gt-label == pass) AND (pass ∉ {p-label})
    - 修正方法：与 gt-box 对应的每个 p-box，相应的 p-label 的过杀加 1。
2. Pass 标签一致
    - 即：(gt-box 与 p-box 有交集) AND (gt-label == pass) AND (pass ∈ {p-label}) AND (len({p-label}) > 1)
    - 修正方法：与 gt-box 对应的每个 p-box，如果为缺陷标签，则相应的 p-label 的过杀加 1。
3. 缺陷标签错检为 pass 标签
    - 即：(gt-box 与 p-box 有交集) AND (gt-label == defect_i) AND (defect_i ∉ {p-label}={pass})
    - 修正方法：缺陷标签 defect_i 的漏检加 1。

### 合并规则

1. `一致`：
    1. 当 `gt-label = defect_i != pass` 时，(gt-box 与 p-box 有交集) AND (gt-label = defect_i ∈ {p-label})
    2. 当 `gt-label == pass` 时。要求 **`gt-box` 所在的图片中 `p-box` 没有 `defect-label`**，且满足以下条件之一：
        - (gt-box 无 p-box 与之有交集)
        - (gt-box 与 p-box 有交集) AND ({p-label} == {pass})
        
2. `漏检`：
    1. 当 `gt-label = defect_i != pass` 时，满足以下条件之一：
        - (gt-box 无 p-box 与之有交集)
        - (gt-box 与 p-box 有交集) AND ({p-label} == {pass})

3. `错检`：
    1. 当 `gt-label = defect_i != pass` 时，要求：(gt-box 与 p-box 有交集) AND (gt-label ∉ {p-label}) AND ({p-label} != {pass})

4. `过杀`：
    1. 当无 gt-box 与 p-box 有交集时。
    2. 当 gt-box 与 p-box 有交集时，(gt-label == pass) AND (p-label != pass)

### 说明

1. 由于一部分的 pass 标签的指标数量被分到 defect-label 中，因此 `一致率 + 漏检率 + 错检率` 可以不为 `100%`
2. 对 pass 标签，`漏检率`、`错检率` 和 `过杀率` 无实义，故全部置为 0。
3. 事实上，通过合并后的计算规则可知，标签级指标的计算结果与 **label 为 pass 的 p-box** 的有无，大小，位置，和数量均无关。

# 开发维护相关

### 数据精度

- 当原始数据的小数点保留位数较少时，`Area` 与 `Length * Width` 之间可能产生一定偏差，此时 `Area` 不应该看作冗余属性
- 例子: `Length = 0.0522, Width = 0.1216, Area = 0.0064`, 此行数据中 `Length * Width = 0.006347520000000001`,
  此时的差值已经超出精度单位 `0.0001` 的一半，即 `round(abs(Length * Width - Area), 4) == 0.0001 > 0`
- 因此，为避免不必要的精度损失
    - 在模型输出过程中（后处理不涉及此部分）：应使用 8 位小数的 `Width` 和 `Length` 计算出 8 位小数的 `Area` 后再截断为 4 位小数。而应该避免先将 `Width` 和 `Length` 截断为 4
      位小数，再计算出 4 位小数的 `Area`
    - 在后处理过程中：在处理 `Area` 属性时，应避免通过 `Length * Width` 重新计算，而是使用原始数据中的 `Area` 属性。

### 性能优化

- `pandas.DataFrame` 的列操作很快，如 `data['z'] = data['x'] * data['y']`；但是行（遍历）操作缓慢，性能损耗很大。在后处理流程中，主要操作为行操作。
- 建议: `pandas.DataFrame` 只作为各个 `procedure 类` 之间用于 **传递数据** 的外部包裹类型。在 `procedure 类` 内部，除了 **列操作** 和 **统计操作（如 sum,
  count）**
  ，其他操作均应避免直接调用 `pandas.DataFrame` 接口

### 性能测试结果:

- CPU: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21 GHz
- 内存: 16GB
- Python: version 3.7
- OS: Windows 10 企业版

| 图片数  | gt 行数 | op 行数 | with_gt 耗时 | without_gt  耗时 |
|------|-------|-------|------------|----------------|
| 138  | 291   | 1529  | 0.62s      | 0.115s         |
| 2530 | 4607  | 11623 | 3.74s      | 0.488s         |
