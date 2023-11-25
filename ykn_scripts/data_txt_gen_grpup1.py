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
    # parser.add_argument("--yolo-dir", type=str, default="/archive/hot4/ykn/dataset/0914BKS")
    parser.add_argument("--yolo-dir", type=str, default="/home/ykn@corp.sse.tongji.edu.cn/efficientteacher/data/quexian_all")
    # parser.add_argument("--yolo-dir", type=str, default="/home/ykn@corp.sse.tongji.edu.cn/efficientteacher/data/group1/Data_20231101")

    parser.add_argument("--yolo-txt", type=str, default="./data/group1/group1_unlabelled_only_is.txt")
    parser.add_argument("--label", type=str, default="false")
    parser.add_argument("--expect",type=str, default="Group_1")

    return parser.parse_args()



# 获取其他的group的数据做成unlabelled的txt
def gen_unlabelled_txt(folder_path,file_txt):
    images=[]

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if '.jpg' in file_path and 'IS' in file_path:
                    images.append(file_path)
    

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

    gen_unlabelled_txt(image_dir,file_txt)