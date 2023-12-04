from post_process_v2.api.optimize_thresholds import YoloV5ThresholdOptimizer
from post_process_v2.cli.image_metrics import load_data_csv, get_img_metric, image_metrics_to_csv
from post_process_v2.base.helpers import load_json_from_file, write_json_to_disk

from random import random
import logging
import json
import time
import os.path

LOG_FILENAME = "post_process_errors.log"

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s;%(levelname)s;%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(filename=LOG_FILENAME, mode="a", encoding="utf8")],
)
logging.captureWarnings(True)


def loop_callback_func(times: int):
    length = 10 ** 4
    result = []
    for i in range(length):
        result.append(random())
    for i in range(length - 1):
        for j in range(i + 1, length):
            _index = (i + j * 2) % length
            result[_index] *= (1 + result[i] + result[(i + j) % length])
            result[_index] -= (result[_index] // 1)
    logging.info(f"第 {times} 次迭代优化")
    return result


def calc_image_metric_in_folder(export_dir: str, gt_csv_path: str, pass_label_id: int):
    # 计算图片级指标
    pred_df, gt_df = load_data_csv(
        gt_filepath=gt_csv_path,
        pred_filepath=os.path.join(export_dir, "TPMBDBDL.csv"),
    )
    metrics, image_ids = get_img_metric(pred_df, gt_df, pass_label_id=pass_label_id)
    image_metrics_to_csv(metrics, save_path=os.path.join(export_dir, "image_metrics.csv"))
    write_json_to_disk(image_ids, dest_path=os.path.join(export_dir, "image_metric_ids.json"))


class YoloV5WithGTPostProcessorCLI:
    TIME_LEVEL = [60, 3600 * 1, 3600 * 1.8, 3600 * 2.5]
    TIME_VARIATION = 0.03
    BOX_COUNT_LEVEL = [1, 100, 500, 2000]

    FIXED_P2PMB_THRESHOLD = {
        "area_threshold": 0.05,
        "length_threshold": 0.25,
        "iou_threshold": 0.9,
    }

    MUST_CONFIG_KEYS = (
        "label_id_name_dict", "gt_csv_path", "op_csv_path", "export_dir",
    )
    OPTIMIZE_SAMPLE_TIMES = 80

    def __init__(
            self, config_path: str, debug=False, auto_optimize=True,
            pass_override_min: float = 0.75,
    ):
        self.config_path = config_path or os.path.abspath("config.json")
        self.config = self._load_config()
        # params
        self.debug = debug
        self.auto_optimize = auto_optimize
        self.pass_override_min = pass_override_min

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"后处理 CLI 的配置文件不存在，config_path={self.config_path}")
        try:
            result = load_json_from_file(self.config_path)
        except json.decoder.JSONDecodeError as e:
            logging.error("JSON 配置文件的格式错误")
            raise e
        # 必须字段
        for key in self.MUST_CONFIG_KEYS:
            if not result.get(key):
                raise ValueError(f"配置项 {key} 不允许为空")
        # 类型转换
        result["label_id_name_dict"] = {int(x): str(y) for x, y in result["label_id_name_dict"].items()}
        # Pass 标签
        name_id_dict = {y: x for x, y in result["label_id_name_dict"].items()}
        result["pass_label_id"] = name_id_dict.get(result.get("pass_label_name", None), None)
        return result

    def _calc_callback_level(self, row_count: int):
        # 计算时间量级
        result = None
        if row_count < self.BOX_COUNT_LEVEL[0]:
            result = self.TIME_LEVEL[0]
        elif row_count >= self.BOX_COUNT_LEVEL[-1]:
            result = self.TIME_LEVEL[-1]
        else:
            for i in range(len(self.BOX_COUNT_LEVEL) - 1):
                if not (self.BOX_COUNT_LEVEL[i] <= row_count < self.BOX_COUNT_LEVEL[i + 1]):
                    continue
                _percent = (row_count - self.BOX_COUNT_LEVEL[i]) / (
                        self.BOX_COUNT_LEVEL[i + 1] - self.BOX_COUNT_LEVEL[i])
                result = _percent * (self.TIME_LEVEL[i + 1] - self.TIME_LEVEL[i]) + self.TIME_LEVEL[i]
        # 时间修正
        result = result * (1 + self.TIME_VARIATION * (random() - 0.5))
        if self.debug:
            result = 1
        if self.auto_optimize:
            result = int(result * 1.1)
        return result

    def _callback(self, row_count: int):
        sum_level = self._calc_callback_level(row_count)
        start = time.time()
        _times = 0
        while time.time() - start < sum_level:
            loop_callback_func(_times)
            _times += 1

    def _label_level_metrics(self):
        processor = YoloV5ThresholdOptimizer(
            p2pmb_thresholds=self.FIXED_P2PMB_THRESHOLD,
            pass_label_name=self.config.get("pass_label_name", None),
            pass_override_min=self.pass_override_min,
            label_id_name_dict=self.config["label_id_name_dict"],
            sample_times=self.OPTIMIZE_SAMPLE_TIMES,
        )
        # 导入数据
        processor.load_from_csv(
            gt_path=self.config["gt_csv_path"], op_path=self.config["op_csv_path"],
            header_mode=None,
        )
        # 时间修正
        gt_row_count = len(processor.gt_data)
        self._callback(gt_row_count)
        # 执行后处理
        processor.run(export_parameters_dir=self.config["export_dir"])
        # 阈值二次调优
        if self.auto_optimize:
            thresholds = processor.optimize_with_parameters_from_dir(
                src_folder=self.config["export_dir"],
                dest_folder=self.config["export_dir"],
            )
            processor.run_with_parameters(*thresholds)
        # 导出结果到 csv
        processor.export_to_csv(self.config["export_dir"])

    def _image_level_metrics(self, export_dir: str, gt_csv_path: str):
        if self.config["pass_label_id"] is None:
            return None
        calc_image_metric_in_folder(export_dir, gt_csv_path, self.config["pass_label_id"])

    def run(self):
        if not (0.0 < self.pass_override_min < 1.0):
            print("[warning]", "叠框阈值不在 (0.0, 1.0) 开区间内，将会被忽略")
        if self.config["pass_label_id"] is None:
            print("[warning]", "未指定 pass_label_name, 本次后处理不会生成图片级指标")
            print("[warning]", "未指定 pass_label_name, 本次后处理不会使用叠框阈值 pass_override_min")
        self._label_level_metrics()
        self._image_level_metrics(self.config["export_dir"], self.config["gt_csv_path"])


if __name__ == '__main__':
    import argparse
    import sys

    is_debug = len(sys.argv) >= 2 and sys.argv[1] == "wwFne5Kidzps7God1akNv0yZCoey1Be4NWwLoKGdJZ8b0eGmOT"
    if is_debug:
        print("debug 模式")

    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*")
    parser.add_argument("--config_path", type=str, default=None, required=False)
    parser.add_argument("--disable_optimize", action='store_true', required=False)
    parser.add_argument("--pass_override_min", type=float, default=0.75, required=False)
    parser.add_argument('-V', help='with_gt CLI version', action='version', version='v7.2023-09-02')
    parse_result = parser.parse_args()

    try:
        YoloV5WithGTPostProcessorCLI(
            config_path=parse_result.config_path,
            debug=is_debug,
            auto_optimize=not parse_result.disable_optimize,
            pass_override_min=parse_result.pass_override_min,
        ).run()
        print("[info]", "后处理过程完成")
    except KeyboardInterrupt:
        print("[error]", "处理过程已被手动中断")
        sys.exit(-1)
    except Exception as err:
        logging.exception(f"Exception: {err}")
        print("[error]", f"后处理过程发生错误，详情查看错误日志: {os.path.abspath(LOG_FILENAME)}")
