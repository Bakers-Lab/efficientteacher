import argparse
from typing import List, Dict, NamedTuple
import os
import shutil
from tools.bks.post_process import EXPORT_TGT_FILENAME, EXPORT_TPMBDBDL_FILENAME
import pandas as pd
import json

ENHACNED_FILENAME = 'gt_enhance_yolo.txt'
PATHS_MAP_FILENAME = 'paths_yolo.txt'


def copy_data_cfg(data_cfg_path: str, postprocess_dir: str) -> None:
    shutil.copy(data_cfg_path, postprocess_dir)


def generate_gt_enhanced_yolo_file(postprocess_dir: str):
    gt_path = os.path.join(postprocess_dir, EXPORT_TGT_FILENAME)
    gt_df = pd.read_csv(gt_path)
    gt_df['LableID'] = gt_df['LableID'] - 1
    gt_enhanced_yolo_filepath = os.path.join(postprocess_dir, ENHACNED_FILENAME)
    gt_df.to_csv(gt_enhanced_yolo_filepath, index=None, header=None)


def generate_tpmbdbdl_txt_file(postprocess_dir: str):
    # TPMBDBDL.csv -> TPMBDBDL.txt
    pred_path = os.path.join(postprocess_dir, EXPORT_TPMBDBDL_FILENAME)
    pred_df = pd.read_csv(pred_path)
    pred_df['LableID'] = pred_df['LableID'] - 1
    new_path = f'{pred_path[:-4]}.txt'
    pred_df.to_csv(new_path, index=None, header=None)


def generate_paths_map_file(postprocess_dir: str, coco_val_json: str, img_path_prefix: str):
    with open(coco_val_json, 'r') as f:
        imgs = json.load(f)['images']
    sorted_imgs = sorted(imgs, key=lambda x: x['id'])
    paths_file_path = os.path.join(postprocess_dir, PATHS_MAP_FILENAME)
    with open(paths_file_path, 'w') as wf:
        for img in sorted_imgs:
            filename = img['file_name']
            img_path = os.path.join(os.path.abspath(img_path_prefix), filename)
            wf.write(f'{img_path}\n')


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare required files for ana.py, primarily artifacts related to the yolov5 framework.")
    parser.add_argument("--data-cfg", type=str, required=True)
    parser.add_argument("--coco-val-json", type=str, required=True)
    parser.add_argument("--postprocess-dir", type=str, required=True)
    parser.add_argument("--img-path-prefix", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    data_cfg = args.data_cfg
    coco_val_json = args.coco_val_json
    postprocess_dir = args.postprocess_dir
    img_path_prefix = args.img_path_prefix
    copy_data_cfg(data_cfg, postprocess_dir)
    generate_gt_enhanced_yolo_file(postprocess_dir)
    generate_paths_map_file(postprocess_dir, coco_val_json, img_path_prefix)
    generate_tpmbdbdl_txt_file(postprocess_dir)