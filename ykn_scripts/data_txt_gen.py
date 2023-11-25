# 在effec teacher给出的样例中, yaml中的给出的是txt格式的数据目录 而现有的yolo格式的目录是直接用目录作为路径的
# 因此需要使用这个脚本进行转化
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
    parser.add_argument("--yolo-dir", type=str, default="/archive/hot4/ykn/dataset/0914BKS")
    parser.add_argument("--yolo-txt", type=str, default="./data/group1/group1_unlabelled.txt")
    parser.add_argument("--label", type=str, default="false")
    parser.add_argument("--expect",type=str, default="Group_1")

    return parser.parse_args()

# 获取一个group的数据 并生成txt
def gen_one_group_txt(image_dir,file_txt):
    file_names=os.listdir(image_dir)
    images=[]
    for file_name in file_names:
        if '.jpg' in file_name:
            images.append(file_name)
    with open(file_txt,"w") as file:
        for image in images:
            image_path=os.path.join(image_dir,image)
            file.write(image_path+"\n")

# 获取其他的group的数据做成unlabelled的txt
def gen_unlabelled_txt(folder_path,file_txt,expect_group):
    images=[]
    file_and_folder_list=os.listdir(folder_path)
    for file_and_folder in file_and_folder_list:
        if 'Group' in file_and_folder and file_and_folder!=expect_group:
            file_folder=os.path.join(folder_path,file_and_folder,"train")
            file_names=os.listdir(file_folder)
            for file_name in file_names:
                if '.jpg' in file_name:
                    images.append(os.path.join(file_folder,file_name))
    with open(file_txt,"w") as file:
        for image in images:
            image_path=os.path.join(image_dir,image)
            file.write(image_path+"\n")
    

if __name__=="__main__":
    # python ykn_scripts/data_txt_gen.py --label true --yolo-dir /archive/hot4/ykn/dataset/0914BKS/Group_1/train --yolo-txt ./data/group1/group1_train.txt
    # python ykn_scripts/data_txt_gen.py --label false --yolo-dir /archive/hot4/ykn/dataset/0914BKS --yolo-txt ./data/group1/group1_unlabelled.txt --expect Group_1
    args=get_args()
    image_dir=args.yolo_dir
    file_txt=args.yolo_txt
    if args.label=='true':
        gen_one_group_txt(image_dir,file_txt)
    else:
        expect_group=args.expect
        gen_unlabelled_txt(image_dir,file_txt,expect_group)