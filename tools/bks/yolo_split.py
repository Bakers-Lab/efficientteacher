import argparse
from pathlib import Path
import os
import random
import json
from typing import Dict, List
import shutil
import yaml
import string


class Spliter(object):

    TRAIN_SUBDIR_NAME = 'train'
    VAL_SUBDIR_NAME = 'val'
    DATA_CFG_FILENAME = 'DATA.yaml'

    def __init__(self, raw_dataet_dir: str, yolo_dataet_dir: str, cls_names: List[str]) -> None:
        self.raw_dataset_dir = raw_dataet_dir
        self.yolo_dataet_dir = yolo_dataet_dir
        self.cls_names = cls_names
        self._init_yolo_dir()

    def _init_yolo_dir(self) -> None:
        # init yolo dir
        if os.path.isdir(self.yolo_dataet_dir):
            shutil.rmtree(self.yolo_dataet_dir)
        os.makedirs(self.yolo_dataet_dir)
        # init train and val subdir
        train_path = os.path.join(self.yolo_dataet_dir, self.TRAIN_SUBDIR_NAME)
        val_path = os.path.join(self.yolo_dataet_dir, self.VAL_SUBDIR_NAME)
        os.makedirs(train_path)
        os.makedirs(val_path)
        # init DATA config
        data_cfg_path = os.path.join(self.yolo_dataet_dir, self.DATA_CFG_FILENAME)
        data_cfg = {
            'names': cls_names,
            'nc': len(cls_names),
            'path': self.yolo_dataet_dir,
            'train': train_path,
            'val': val_path,
            'test': val_path
        }
        with open(data_cfg_path, 'w') as wf:
            yaml.safe_dump(data_cfg, wf)

    def _generate_random_prefix(self, prefix_length: int):
        characters = string.ascii_letters + string.digits
        prefix = ''.join(random.choice(characters) for _ in range(prefix_length))
        return prefix

    def replace_extension(sefl, file_path, new_extension):
        return file_path[:-4] + new_extension

    def split(self) -> None:
        yolo_path = Path(self.raw_dataset_dir)
        img_paths = sorted(yolo_path.rglob("*.jpg"))
        img_paths += sorted(yolo_path.rglob("*.png"))
        random.shuffle(img_paths)
        train_image_paths = img_paths[:int(len(img_paths) * 0.7)]
        val_image_paths = img_paths[int(len(img_paths) * 0.7):]
        # copy images one bye one to avoid same filename
        for img_path in train_image_paths:
            filename = os.path.basename(img_path)
            new_filename = f'{self._generate_random_prefix(5)}_{filename}'
            new_path = os.path.join(*[self.yolo_dataet_dir, self.TRAIN_SUBDIR_NAME, new_filename])
            shutil.copy(img_path, new_path)
            label_filepath = self.replace_extension(str(img_path), '.txt')
            new_label_filepath = self.replace_extension(str(new_path), '.txt')
            shutil.copy(label_filepath, new_label_filepath)
        for img_path in val_image_paths:
            filename = os.path.basename(img_path)
            new_filename = f'{self._generate_random_prefix(5)}_{filename}'
            new_path = os.path.join(*[self.yolo_dataet_dir, self.VAL_SUBDIR_NAME, new_filename])
            shutil.copy(img_path, new_path)
            label_filepath = self.replace_extension(str(img_path), '.txt')
            new_label_filepath = self.replace_extension(str(new_path), '.txt')
            shutil.copy(label_filepath, new_label_filepath)


def get_args():
    parser = argparse.ArgumentParser(description="Execute yolov5 dataset spliting parameters")
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--yolo", type=str, required=True)
    parser.add_argument('--cls-names',
                        metavar='cls_names',
                        type=str,
                        nargs='+',
                        required=True,
                        help='List of class names')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    raw_data_dir = args.raw
    yolo_data_dir = args.yolo
    cls_names = args.cls_names
    spliter = Spliter(raw_data_dir, yolo_data_dir, cls_names)
    spliter.split()
