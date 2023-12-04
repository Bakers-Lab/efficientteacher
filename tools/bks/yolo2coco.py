import argparse
import os
from pathlib import Path
import imagesize
import json
from typing import Dict, List, Tuple
import random
import shutil
import yaml


def create_image_annotation(file_path: Path, width: int, height: int, image_id: int):
    file_path = file_path.name
    image_annotation = {
        "file_name": file_path,
        "height": height,
        "width": width,
        "id": image_id,
    }
    return image_annotation


def create_annotation_from_yolo_format(min_x,
                                       min_y,
                                       width,
                                       height,
                                       image_id,
                                       category_id,
                                       annotation_id,
                                       segmentation=True):
    bbox = (float(min_x), float(min_y), float(width), float(height))
    area = width * height
    max_x = min_x + width
    max_y = min_y + height
    if segmentation:
        seg = [[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]]
    else:
        seg = []
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id,
        "segmentation": seg,
    }

    return annotation


class Yolo2CocoConverter(object):
    """
    COCO Dataset structure:
    .
    ├── annotations
    │   ├── train.json
    │   └── val.json
    ├── test
    ├── train
    └── val
    """
    TRAIN_SUBDIR_NAME = 'train'
    VAL_SUBDIR_NAME = 'val'
    TEST_SUBDIR_NAME = 'test'
    TRAIN_ANN_FILENAME = 'train.json'
    VAL_ANN_FILENAME = 'val.json'

    def __init__(self, yolo_dataset_cfg: str, coco_dataset_root: str) -> None:
        self.yolo_dataset_cfg = yolo_dataset_cfg
        self.coco_dataset_root = coco_dataset_root
        self._load_yolo_data_cfg()

    def _load_yolo_data_cfg(self) -> None:
        with open(self.yolo_dataset_cfg, 'r') as f:
            data_cfg = yaml.safe_load(f)
        self.yolo_dataset_root = data_cfg['path']
        self.yolo_train_dir = data_cfg['train']
        self.yolo_val_dir = data_cfg['val']
        self.yolo_test_dir = data_cfg['test']
        self.cls_names = data_cfg['names']

    @property
    def coco_train_ann_file_path(self):
        return os.path.join(self.coco_dataset_root, self.TRAIN_ANN_FILENAME)

    @property
    def coco_val_ann_file_path(self):
        return os.path.join(self.coco_dataset_root, self.VAL_ANN_FILENAME)

    @property
    def coco_train_dir(self):
        return os.path.join(self.coco_dataset_root, self.TRAIN_SUBDIR_NAME)

    @property
    def coco_val_dir(self):
        return os.path.join(self.coco_dataset_root, self.VAL_SUBDIR_NAME)

    def _gen_annotation_file(self, img_paths: List[str], box2seg=False) -> Dict:
        # check coco train dir and coco val dir
        if not os.path.isdir(self.coco_train_dir) or not os.path.isdir(self.coco_val_dir):
            raise ValueError(f'coco train dir or coco val dir doesn\'t exist')

        annotations = []
        images_annotations = []

        image_id = 0
        annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'
        for img_path in img_paths:
            file_path = Path(img_path)
            # Check how many items have progressed
            print(f'\rProcessing + {str(image_id)} ...', end='')
            # Build image annotation, known the image's width and height
            w, h = imagesize.get(str(file_path))
            image_annotation = create_image_annotation(file_path=file_path, width=w, height=h, image_id=image_id)
            images_annotations.append(image_annotation)
            yolo_label_file_path = Path(f'{file_path.parents[0]}/{file_path.stem}.txt')
            if not Path(yolo_label_file_path).exists():
                raise ValueError(f'yolo label file:{yolo_label_file_path} doesn\'t exist.'
                                )  # The image may not have any applicable annotation txt file.
            with open(yolo_label_file_path, 'r') as label_file:
                label_lines = label_file.readlines()
                # yolo format - (class_id, x_center, y_center, width, height)
                # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
            for label_line in label_lines:
                category_id = (int(label_line.split()[0]) + 1)  # you start with annotation id with '1'
                x_center = float(label_line.split()[1])
                y_center = float(label_line.split()[2])
                width = float(label_line.split()[3])
                height = float(label_line.split()[4])

                float_x_center = w * x_center
                float_y_center = h * y_center
                float_width = w * width
                float_height = h * height

                min_x = int(float_x_center - float_width / 2)
                min_y = int(float_y_center - float_height / 2)
                width = int(float_width)
                height = int(float_height)

                annotation = create_annotation_from_yolo_format(
                    min_x,
                    min_y,
                    width,
                    height,
                    image_id,
                    category_id,
                    annotation_id,
                    segmentation=box2seg,
                )
                annotations.append(annotation)
                annotation_id += 1
            image_id += 1
        coco_ann = {'images': images_annotations, 'annotations': annotations, 'categories': []}
        return coco_ann

    def _copy_imgs(self) -> Tuple[List[str], List[str]]:
        yolo_train_path = Path(self.yolo_train_dir)
        yolo_train_img_paths = sorted(yolo_train_path.rglob("*.jpg")) + sorted(yolo_train_path.rglob("*.png"))
        yolo_val_path = Path(self.yolo_val_dir)
        yolo_val_img_paths = sorted(yolo_val_path.rglob("*.jpg")) + sorted(yolo_val_path.rglob("*.png"))
        if os.path.isdir(self.coco_train_dir):
            shutil.rmtree(self.coco_train_dir)
        os.makedirs(self.coco_train_dir)
        if os.path.isdir(self.coco_val_dir):
            shutil.rmtree(self.coco_val_dir)
        os.makedirs(self.coco_val_dir)
        for img_path in yolo_train_img_paths:
            shutil.copy(img_path, self.coco_train_dir)
        for img_path in yolo_val_img_paths:
            shutil.copy(img_path, self.coco_val_dir)
        return yolo_train_img_paths, yolo_val_img_paths

    def _copy_yolo_cfg(self) -> None:
        new_cfg_path = os.path.join(self.coco_dataset_root, os.path.basename(self.yolo_dataset_cfg))
        shutil.copy(self.yolo_dataset_cfg, new_cfg_path)

    def convert(self) -> None:
        self._copy_yolo_cfg()
        yolo_train_imgs_path, yolo_val_imgs_path = self._copy_imgs()
        train_coco_ann = self._gen_annotation_file(yolo_train_imgs_path)
        val_coco_ann = self._gen_annotation_file(yolo_val_imgs_path)
        for index, label in enumerate(self.cls_names):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            train_coco_ann["categories"].append(categories)
            val_coco_ann["categories"].append(categories)
        if not os.path.exists(self.coco_dataset_root):
            os.makedirs(self.coco_dataset_root)
        with open(self.coco_train_ann_file_path, 'w') as wf:
            json.dump(train_coco_ann, wf, indent=4)
        with open(self.coco_val_ann_file_path, 'w') as wf:
            json.dump(val_coco_ann, wf, indent=4)


def get_args():
    parser = argparse.ArgumentParser(description="Execute yolov5 dataset spliting parameters")
    parser.add_argument("--coco-dir", type=str, required=True)
    parser.add_argument("--yolo-cfg", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    yolo_data_cfg = args.yolo_cfg
    coco_dataset_root = args.coco_dir
    converter = Yolo2CocoConverter(yolo_data_cfg, coco_dataset_root)
    converter.convert()
