from post_process_v2.cli.with_gt import YoloV5WithGTPostProcessorCLI, LOG_FILENAME
from post_process_v2.api.with_gt import YoloV5WithGTPostProcessor
from post_process_v2.procedures.init_parameters import Threshold
from post_process_v2.base.helpers import load_json_from_file

import logging
import os.path


class YoloV5WithGTParametersPostProcessorCLI(YoloV5WithGTPostProcessorCLI):
    MUST_CONFIG_KEYS = ("label_id_name_dict",)
    CALLBACK_LEVEL_RATIO = 0.4
    PARAMETER_KEYS = ("op2p", "pmbdb2pmbdbdl")

    DEFAULT_OPTIMIZE_FOLDER = "optimization"
    DEFAULT_GT_CSV_PATH = "gt_enhance_yolo.txt"
    DEFAULT_OP_CSV_PATH = "pred_enhance_yolo.txt"
    DEFAULT_CONFIG_PATH = "Postprocess.json"

    def __init__(
            self, work_root: str, config_path: str = None, optimize_folder: str = None,
            gt_csv_path: str = None, op_csv_path: str = None, debug=False,
    ):
        YoloV5WithGTPostProcessorCLI.__init__(
            self, debug=debug,
            config_path=os.path.join(work_root, config_path or self.DEFAULT_CONFIG_PATH),
        )
        self.work_root = work_root
        # 可选路径
        self.optimize_folder = os.path.join(work_root, optimize_folder or self.DEFAULT_OPTIMIZE_FOLDER)
        self.gt_csv_path = os.path.join(work_root, gt_csv_path or self.DEFAULT_GT_CSV_PATH)
        self.op_csv_path = os.path.join(work_root, op_csv_path or self.DEFAULT_OP_CSV_PATH)

    def _calc_callback_level(self, row_count: int):
        return super()._calc_callback_level(row_count) * self.CALLBACK_LEVEL_RATIO

    '''
    protected: 数据校验
    '''

    def _verify_thresholds(self):
        result = dict()
        for _key in self.PARAMETER_KEYS:
            _path = os.path.join(self.optimize_folder, f"{_key}_parameters.json")
            if not os.path.exists(_path):
                raise FileNotFoundError(f"阈值文件不存在, path={_path}")
            result[f"{_key}_config"] = {str(x): Threshold(**y) for x, y in load_json_from_file(_path).items()}
        return result

    def _verify_data_files(self):
        for _path in [self.gt_csv_path, self.op_csv_path]:
            if os.path.exists(_path):
                continue
            raise FileNotFoundError(f"指定的数据文件不存在: path={_path}")

    '''
    protected: 计算指标
    '''

    def _label_level_metrics(self):
        processor = YoloV5WithGTPostProcessor(
            p2pmb_thresholds=self.FIXED_P2PMB_THRESHOLD,
            pass_label_name=self.config.get("pass_label_name", None),
            label_id_name_dict=self.config["label_id_name_dict"],
        )
        # 导入数据
        processor.load_from_csv(gt_path=self.gt_csv_path, op_path=self.op_csv_path, header_mode=None)
        # 时间修正
        gt_row_count = len(processor.gt_data)
        self._callback(gt_row_count)
        # 执行后处理
        processor.run_with_parameters_from_dir(self.optimize_folder)
        # 导出结果到 csv
        processor.export_to_csv(self.optimize_folder)

    '''
    public
    '''

    def run(self):
        self._verify_thresholds()
        self._verify_data_files()
        if self.config["pass_label_id"] is None:
            print("[warning]", "未指定 pass_label_name, 本次后处理不会生成图片级指标")
        self._label_level_metrics()
        self._image_level_metrics(self.optimize_folder, self.gt_csv_path)


if __name__ == '__main__':
    import argparse
    import sys

    is_debug = len(sys.argv) >= 2 and sys.argv[1] == "wwFne5Kidzps7God1akNv0yZCoey1Be4NWwLoKGdJZ8b0eGmOT"
    if is_debug:
        print("debug 模式")

    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*")
    parser.add_argument("--work_root", type=str, default=None, required=True)
    parser.add_argument("--config_path", type=str, default=None, required=False)
    parser.add_argument("--optimize_folder", type=str, default=None, required=False)
    parser.add_argument("--gt_csv_path", type=str, default=None, required=False)
    parser.add_argument("--op_csv_path", type=str, default=None, required=False)
    parse_result = parser.parse_args()

    try:
        YoloV5WithGTParametersPostProcessorCLI(
            work_root=parse_result.work_root,
            config_path=parse_result.config_path,
            optimize_folder=parse_result.optimize_folder,
            gt_csv_path=parse_result.gt_csv_path,
            op_csv_path=parse_result.op_csv_path,
            debug=is_debug,
        ).run()
        print("[info]", "后处理过程完成")
    except KeyboardInterrupt:
        print("[error]", "处理过程已被手动中断")
        sys.exit(-1)
    except Exception as err:
        logging.exception(f"Exception: {err}")
        print("[error]", f"后处理过程发生错误，详情查看错误日志: {os.path.abspath(LOG_FILENAME)}")
