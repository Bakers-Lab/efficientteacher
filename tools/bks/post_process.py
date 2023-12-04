import json
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import os
import argparse
from post_process_v2.api.without_gt import YoloV5WithoutGTPostProcessor
from post_process_v2.api.with_gt import YoloV5WithGTPostProcessor
from post_process_v2.api.optimize_thresholds import YoloV5ThresholdOptimizer
import logging

PREDS_COLUMN_NAMS = [
    "CenterX",
    "CenterY",
    "Length",
    "Width",
    "Confidence",
    "ImageID",
    "LableID",
    "Area",
    "BoxID",
]
GT_COLUMNS_NAMES = [
    "CenterX",
    "CenterY",
    "Length",
    "Width",
    "ImageID",
    "LableID",
    "Area",
    "BoxID",
]

# file
PATHS_FILENAME = 'paths_yolo.csv'
GT_FILENAME = 'gt_enhance_yolo.csv'
PRED_FILENAME = 'pred.csv'
BKS_DIRNAME = 'BKS'
OPTIM_DIRNAME = 'OPTIM'
RAW_PARM_DIRNAME = 'RAW'
DATA_CONFIG_NAME = 'DATA.yaml'

# export
EXPORT_DIRNAME = 'POST_PROCESSED'
EXPORT_TP_FILENAME = 'TP.csv'
EXPORT_TPMB_FILENAME = 'TPMB.csv'
EXPORT_TPMBDB_FILENAME = 'TPMBDB.csv'
EXPORT_TPMBDBDL_FILENAME = 'TPMBDBDL.csv'
EXPORT_TGT_FILENAME = 'TGT.csv'
EXPORT_RESULT_FILENAME = 'RESULT.csv'


@dataclass
class IMG_SIZE:
    height: str
    width: str


class PostProcesser(object):

    def __init__(self) -> None:
        pass

    def _pred_dict2df(self, pred_dicts: List[Dict], img_size_map: Dict[int, IMG_SIZE]) -> pd.DataFrame:
        """
        [
            {
                "image_id": 0,
                "bbox": [
                    2.53248929977417,
                    33.98996353149414,
                    72.1782956123352,
                    46.141788482666016
                ],
                "score": 0.9579349756240845,
                "category_id": 3
            },
        ]
        """
        bbox_list = []
        for bbox_id, pred in enumerate(pred_dicts):
            img_id = pred['image_id']
            img_w = img_size_map[img_id].width
            img_h = img_size_map[img_id].height
            w = pred['bbox'][2] / img_w
            h = pred['bbox'][3] / img_h
            w = 1e-6 if w == 0.0 else w
            h = 1e-6 if h == 0.0 else h
            bbox = {
                'CenterX': (pred['bbox'][0] + pred['bbox'][2] / 2) / img_w,
                'CenterY': (pred['bbox'][1] + pred['bbox'][3] / 2) / img_h,
                'Length': w,
                'Width': h,
                'Confidence': pred['score'],
                'ImageID': img_id,
                'LableID': pred['category_id'],
                'Area': w * h,
                'BoxID': bbox_id
            }
            bbox_list.append(bbox)
            if w * h == 0 or bbox_id == 2004:
                print(bbox)
                print(pred)

        df = pd.DataFrame(bbox_list)
        print(df)
        return df

    def _gt_dict2df(self, ann_dicts: List[Dict], img_size_map: Dict[int, IMG_SIZE]) -> pd.DataFrame:
        bbox_list = []
        for bbox_id, gt in enumerate(ann_dicts):
            img_id = gt['image_id']
            img_w = img_size_map[img_id].width
            img_h = img_size_map[img_id].height
            # print('*' * 80)
            # print(img_size_map)
            # print(img_w, img_h, gt['bbox'])
            w = gt['bbox'][2] / img_w
            h = gt['bbox'][3] / img_h
            bbox = {
                'CenterX': (gt['bbox'][0] + gt['bbox'][2] / 2) / img_w,
                'CenterY': (gt['bbox'][1] + gt['bbox'][3] / 2) / img_h,
                'Length': w,
                'Width': h,
                'ImageID': img_id,
                'LableID': gt['category_id'],
                'Area': w * h,
                'BoxID': bbox_id
            }
            bbox_list.append(bbox)
        df = pd.DataFrame(bbox_list)
        print(df)
        return df

    def process(self, pred_dicts: List[Dict], ann_dicts: List[Dict], img_size_map: Dict[int, IMG_SIZE],
                categories: Dict, export_dir: str):

        if not os.path.isdir(export_dir):
            os.makedirs(export_dir)
        print(export_dir)
        p2pmb_thresholds = {
            "area_threshold": 0.05,
            "length_threshold": 0.25,
            "iou_threshold": 0.9,
        }
        pred_df = self._pred_dict2df(pred_dicts, img_size_map)
        gt_df = self._gt_dict2df(ann_dicts, img_size_map)
        pred_df.to_csv('pred.csv')
        gt_df.to_csv('gt.csv')
        label_id2name = self._categories2label_id2name(categories)

        processer = YoloV5ThresholdOptimizer(label_id_name_dict=label_id2name,
                                             p2pmb_thresholds=p2pmb_thresholds,
                                             pass_label_name='PASS'
                                             )
        processer.load_from_dataframe(gt_df, pred_df)
        processer.run(export_parameters_dir=os.path.join(export_dir, RAW_PARM_DIRNAME))
        optim_dir = os.path.join(export_dir, OPTIM_DIRNAME)
        processer.optimize_with_parameters_from_dir(src_folder=os.path.join(export_dir, RAW_PARM_DIRNAME),
                                                    dest_folder=optim_dir)
        processer.run_with_parameters_from_dir(optim_dir)
        processer.export_to_csv(dest_dir=export_dir, with_input_data=True)
        # report_df.to_csv(os.path.join(export_dir, EXPORT_RESULT_FILENAME))
        # output_df = processer.export_to_csv(export_dir, with_header=True)

    def _categories2label_id2name(self, categories: List[Dict]) -> Dict[str, str]:
        label_id2name = {}
        for cat in categories:
            label_id2name[cat['id']] = cat['name']
        return label_id2name


def get_args():
    parser = argparse.ArgumentParser(description="Execute post-processing parameters")
    parser.add_argument("--pred-json", type=str, required=True)
    parser.add_argument("--coco-val-json", type=str, required=True)
    parser.add_argument("--coco-val-dir", type=str, required=True)
    parser.add_argument("--export-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='post_process.log',
                        filemode='w')
    args = get_args()
    result_json_file = args.pred_json
    coco_ann_file_path = args.coco_val_json
    # result_json_file = './wzj-debug-01.bbox.json'
    # coco_ann_file_path = './data/DOCK_L0/new_val.json'
    coco_val_dir = args.coco_val_dir
    export_dir = args.export_dir
    # load pred_dicts
    with open(result_json_file, 'r') as f:
        pred_dicts = json.load(f)
    # load img size dict
    with open(coco_ann_file_path, 'r') as f:
        coco_ann = json.load(f)
        img_size_map = {}
        for img in coco_ann['images']:
            img_size_map[img['id']] = IMG_SIZE(width=img['width'], height=img['height'])
        ann_dicts = coco_ann['annotations']
        categories = coco_ann['categories']
    post_processer = PostProcesser()
    # execute post_process
    post_processer.process(pred_dicts, ann_dicts, img_size_map, categories, export_dir)
