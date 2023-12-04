from post_process_v2.cli.with_gt import YoloV5WithGTPostProcessorCLI, LOG_FILENAME
from post_process_v2.api.without_gt import YoloV5WithoutGTPostProcessor

import logging
import os.path


class YoloV5WithoutGTPostProcessorCLI(YoloV5WithGTPostProcessorCLI):
    MUST_CONFIG_KEYS = ("label_id_name_dict",)

    DEFAULT_CONFIG_PATH = "Postprocess.json"
    DEFAULT_OP_CSV_PATH = "pred_enhance_yolo.txt"

    PARAMETER_KEYS = ("op2p", "pmbdb2pmbdbdl")

    def __init__(
            self, work_dir: str, threshold_dir: str, export_dir: str,
            op_csv_path: str, config_path: str, debug=False, pass_override_min: float = 0.75,
    ):
        YoloV5WithGTPostProcessorCLI.__init__(
            self, config_path=os.path.join(work_dir, config_path),
            debug=debug, auto_optimize=False, pass_override_min=pass_override_min,
        )
        self.work_dir = work_dir
        self.threshold_dir = threshold_dir
        self.export_dir = export_dir
        self.op_csv_path = self.__build_op_csv_path(op_csv_path)

    '''
    private
    '''

    def __build_op_csv_path(self, op_csv_path):
        # 必须存在的路径
        exist_paths = [self.work_dir, self.threshold_dir, os.path.join(self.work_dir, op_csv_path)]
        exist_paths.extend([os.path.join(self.threshold_dir, f"{x}_parameters.json") for x in self.PARAMETER_KEYS])
        # 校验
        for _path in exist_paths:
            if not os.path.exists(_path):
                raise FileNotFoundError(_path)
        return os.path.join(self.work_dir, op_csv_path)

    '''
    protected
    '''

    def _calc_callback_level(self, row_count: int):
        return 0 if self.debug else 1

    def _op_to_pmbdbdl(self):
        processor = YoloV5WithoutGTPostProcessor(
            p2pmb_thresholds=self.FIXED_P2PMB_THRESHOLD,
            label_id_name_dict=self.config["label_id_name_dict"],
            pass_label_name=self.config["pass_label_name"],
            pass_override_min=self.pass_override_min,
        )
        # 导入数据
        processor.load_from_csv(self.op_csv_path, header_mode=None)
        # 时间修正
        op_row_count = len(processor.op_data)
        self._callback(op_row_count)
        # 执行后处理
        processor.run_with_parameter_files(self.threshold_dir)
        # 导出结果到 csv
        processor.export_to_csv(self.export_dir)

    '''
    public
    '''

    def run(self):
        if not (0.0 < self.pass_override_min < 1.0):
            print("[warning]", "叠框阈值不在 (0.0, 1.0) 开区间内，将会被忽略")
        if self.config["pass_label_id"] is None:
            print("[warning]", "未指定 pass_label_name, 本次后处理不会使用叠框阈值 pass_override_min")
        self._op_to_pmbdbdl()


if __name__ == '__main__':
    import argparse
    import sys

    is_debug = len(sys.argv) >= 2 and sys.argv[1] == "BQKxnu2EyZ2aKIhwaHyZRoF9QN4C3nL6vFkw9Ha03XNsVz5NJ9"
    if is_debug:
        print("debug 模式")

    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*")
    # 必须参数
    parser.add_argument("--bks_dir", type=str, required=True)
    parser.add_argument("--threshold_dir", type=str, required=True)
    parser.add_argument("--export_dir", type=str, required=True)
    # 可选参数
    parser.add_argument(
        "--op_csv_path", type=str, required=False,
        default=YoloV5WithoutGTPostProcessorCLI.DEFAULT_OP_CSV_PATH,
    )
    parser.add_argument(
        "--config_path", type=str, required=False,
        default=YoloV5WithoutGTPostProcessorCLI.DEFAULT_CONFIG_PATH,
    )
    parser.add_argument("--pass_override_min", type=float, default=0.75, required=False)
    parser.add_argument('-V', help='without_gt CLI version', action='version', version='v4.2023-09-02')
    parse_result = parser.parse_args()

    try:
        YoloV5WithoutGTPostProcessorCLI(
            work_dir=parse_result.bks_dir,
            threshold_dir=parse_result.threshold_dir,
            export_dir=parse_result.export_dir,
            config_path=parse_result.config_path,
            op_csv_path=parse_result.op_csv_path,
            pass_override_min=parse_result.pass_override_min,
            debug=is_debug,
        ).run()
        print("[info]", "不包含 GT 数据的后处理过程完成")
    except KeyboardInterrupt:
        print("[error]", "处理过程已被手动中断")
        sys.exit(-1)
    except Exception as err:
        logging.exception(f"Exception: {err}")
        print("[error]", f"不包含 GT 数据的后处理过程发生错误，详情查看错误日志: {os.path.abspath(LOG_FILENAME)}")
