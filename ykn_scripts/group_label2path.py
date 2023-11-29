import argparse
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple
import random
import shutil
import yaml

def get_args():
    # yolo dir指的是存放原始的文件的
    parser = argparse.ArgumentParser(description="Execute yolov5 dataset spliting parameters")
    
    parser.add_argument("--root_path", type=str, default="/archive/hot4/ykn/dataset/0914BKS")
    parser.add_argument("--target_path", type=str, default="/archive/hot4/ykn/dataset/Group_label_path")

    return parser.parse_args()

def read_labels(path: str):
    with open(path,"r") as file:
        labels_lines=file.readlines()
    labels=[label_line.strip() for label_line in labels_lines]
    return labels

if __name__=='__main__':
    args=get_args()
    root_path=args.root_path
    target_path=args.target_path
    group_folder_list=os.listdir(root_path)
    for group_folder in group_folder_list:
        if 'Group' in group_folder:
            label_path=os.path.join(root_path,group_folder,"classes.txt")
            label_mapping=read_labels(label_path)
            img_folder=os.path.join(root_path,group_folder,"train")
            raw_files=os.listdir(img_folder)
            imgs=[file for file in raw_files if file.endswith('.jpg')]
            for img_name in imgs:
                label_file_name = img_name.replace('.jpg', '.txt')
                with open(os.path.join(img_folder,label_file_name),"r") as label_content:
                    label_lines=label_content.readlines()
                    label_id=0
                    for label_line in label_lines:
                        label_id = int(label_line.split()[0])
                    label_str=label_mapping[label_id]
                    new_img_folder=os.path.join(target_path,group_folder,label_str)
                    if not os.path.exists(new_img_folder):
                        os.makedirs(new_img_folder)
                    new_img_path=os.path.join(new_img_folder,img_name)
                    shutil.copyfile(os.path.join(img_folder,img_name), new_img_path)
