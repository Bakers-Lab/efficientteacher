from post_process_v2.api.base import YoloV5PostProcessorBase
from post_process_v2.api.components.with_gt_import_export_trait import WithGTImportExportTrait

from post_process_v2.procedures.components.base_p_procedure import BaseProcedure
from post_process_v2.procedures.init_parameters import Threshold, init_parameters
from post_process_v2.procedures.match_with_gt import PostProcessMatchWithGT
from post_process_v2.procedures.report_ratio import PostProcessReportRatio
from post_process_v2.procedures.analysis_data import PostProcessAnalysisData

from typing import Tuple, List, Iterator, Dict
import pandas as pd
import logging
import time


class YoloV5WithGTPostProcessor(WithGTImportExportTrait, YoloV5PostProcessorBase):
    """
    包含 GT 数据的后处理接口类

    方式一: 自动计算 `阈值`，共两轮的后处理过程 (run 方法)
    输入: OP 数据 + GT 数据
    输出: `阈值` + ratio

    方式二: 给定 `阈值`，共一轮的后处理过程 (run_with_parameters 或 run_with_parameters_from_dir 方法)
    输入: OP 数据 + GT 数据 + `阈值`
    输出: ratio

    其中，上述两种方式中的 `阈值`，具体指 op2p 和 pmbdb2pmbdbdl 两个阶段的阈值参数
    """

    PROCESS_KEYS_FOR_ANALYSIS = ("OP", "PMBDB")

    def __init__(
            self, label_id_name_dict: Dict[int, str], p2pmb_thresholds: Dict[str, float],
            pass_label_name: str = None, pass_override_min: float = 0.75,
    ):
        WithGTImportExportTrait.__init__(self)
        YoloV5PostProcessorBase.__init__(
            self, label_id_name_dict, p2pmb_thresholds, pass_label_name, pass_override_min,
        )
        # values
        self.gt_data = None

    '''
    private
    '''

    def _build_p_map_gt_data(self, op_process_list: List[Tuple[str, BaseProcedure]]) -> \
            Tuple[List[Tuple[str, pd.DataFrame]], List[Tuple[str, pd.DataFrame]]]:
        # 获取各处理阶段的数据
        p_data_list = [("OP", self.op_data)]
        p_data_list.extend([(_name, p.result) for _name, p in op_process_list])
        # 执行匹配
        gt_data_list = [
            (_name, PostProcessMatchWithGT(self.gt_data, _data).run(self.pass_label_id))
            for _name, _data in p_data_list
        ]
        return p_data_list, gt_data_list

    '''
    protected
    '''

    def _input_data_for_export(self) -> Iterator[Tuple[str, pd.DataFrame, List[str]]]:
        name_list = ["OP", "GT"]
        data_list = [self.op_data, self.gt_data]
        columns_list = [self.DEFAULT_OP_COLUMNS, self.DEFAULT_GT_COLUMNS]
        return zip(name_list, data_list, columns_list)

    def _calc_ratio(self, p_data_list: List[Tuple[str, pd.DataFrame]], map_data_list: List[Tuple[str, pd.DataFrame]]):
        self.process_list.append((
            "R4Ratio",
            PostProcessReportRatio(self.label_id_name_dict, self.pass_label_name, p_data_list, map_data_list).run(),
        ))
        logging.info("处理流程结束: report ratio")

    def _run_by_parameters(
            self, op2p_config: Dict[str, Threshold],
            pmbdb2pmbdbdl_config: Dict[str, Threshold],
            with_ratio=True, with_analysis_data=True,
    ) -> Dict[str, pd.DataFrame]:
        # ========== Step1: 处理 op 数据 ==========
        self.process_list = self._op_processes(op2p_config, pmbdb2pmbdbdl_config)
        # ========== Step2: 处理 gt 数据 ==========
        p_data_list, map_data_list = self._build_p_map_gt_data(self.process_list)
        if with_ratio:
            self._calc_ratio(p_data_list, map_data_list)
        # ========== Step3: 构建用于 analysis 的数据 ==========
        result = dict()
        for key, p_data in p_data_list:
            if not (with_analysis_data and key in self.PROCESS_KEYS_FOR_ANALYSIS):
                continue
            # 用于计算阈值的 map 表，不用忽略 pass 标签
            result[f"{key.lower()}_data"] = PostProcessMatchWithGT(self.gt_data, p_data).run()
        return result

    '''
    public: 输入
    '''

    # 父类方法: load_from_numpy

    # 父类方法: load_from_dataframe

    # 父类方法: load_from_csv

    # 父类方法: load_from_db

    '''
    public: 输出
    '''

    # 父类方法: export_to_csv

    # 父类方法: export_to_db

    # 父类方法: export_ratio_to_dataframe

    '''
    public: run
    '''

    def run(self, export_parameters_dir: str):
        """
        先后采用 `置空的阈值` 和 `计算出的阈值` 进行共计两轮的后处理过程
        :param export_parameters_dir: 将 `计算出的阈值` 写入此文件夹。包含 op2p_parameters.json 和 pmbdb2pmbdbdl_parameters.json 两个文件
        :return: self
        """
        self._verify_dataframe_columns()
        time_start = time.time()
        logging.info("[start] 开始: 包含 gt 数据的后处理流程（自动计算阈值方式）")
        # 使用初始化（全部置零）的参数运行
        data = self._run_by_parameters(
            op2p_config=init_parameters(self.label_id_name_dict),
            pmbdb2pmbdbdl_config=init_parameters(self.label_id_name_dict),
            with_ratio=False, with_analysis_data=True,
        )
        # 根据运行结果，计算参数
        analyzer = PostProcessAnalysisData(self.label_id_name_dict, **data).run()
        analyzer.export_to_json(export_parameters_dir)
        # 使用计算后的参数运行
        self._run_by_parameters(*analyzer.result, with_analysis_data=False)
        logging.info(f"[end] 结束: 包含 gt 数据的后处理流程（自动计算阈值方式），耗时 {time.time() - time_start:.2f}s")
        return self

    def run_with_parameters(
            self, op2p_config: Dict[str, Threshold],
            pmbdb2pmbdbdl_config: Dict[str, Threshold],
    ):
        """
        使用给定的阈值参数，进行共计一轮的后处理过程
        :param op2p_config: 字典类型。key 为 label-name, value 为 Threshold 类对象
        :param pmbdb2pmbdbdl_config: 字典类型。key 为 label-name, value 为 Threshold 类对象
        :return: self
        """
        self._verify_dataframe_columns()
        time_start = time.time()
        logging.info("[start] 开始: 包含 gt 数据的后处理流程（指定阈值方式）")
        self._run_by_parameters(
            op2p_config, pmbdb2pmbdbdl_config,
            with_ratio=True, with_analysis_data=False,
        )
        logging.info(f"[end] 结束: 包含 gt 数据的后处理流程（指定阈值方式），耗时 {time.time() - time_start:.2f}s")
        return self

    def run_with_parameters_from_dir(self, folder: str):
        # 指定的参数文件夹下，应包含以下两个文件:
        # op2p_parameters.json
        # pmbdb2pmbdbdl_parameters.json
        return self.run_with_parameters(
            **self._load_parameter_dir_files(folder),
        )
