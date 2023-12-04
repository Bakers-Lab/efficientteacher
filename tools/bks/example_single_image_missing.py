import argparse
import json
import os
import pandas as pd

def load_test_image_from_json(path:str)->list:
    with open(path,'r') as json_file:
        json_dict=json.load(json_file)
    images=json_dict['images']
    image_list=[]
    for image in images:
        image_list.append(image['file_name'])
    return image_list

def load_missing_result(path:str,missing_dict)->list:
    with open(path,"r") as missing_file:
        lines=missing_file.readlines()
        for line in lines:
            image_file_name=line.strip().split(',')[3]
            missing_dict[image_file_name]+=1
    return missing_dict
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', type=str, help='export dir', required=True)
#     args = parser.parse_args()
#     return args


if __name__=="__main__":
    test_json_path=os.path.join("data","rb_ins","test.json")
    seg_missing_path=os.path.join("work_dirs","mask2former_ykn","export-01","missing.txt")
    v5_missing_path=os.path.join("work_dirs","yolov5_s-v61_syncbn_8xb16-300e_ykn","export-01","missing.txt")
    v8_missing_path=os.path.join("work_dirs","yolov8_s_syncbn_fast_8xb16-500e_ykn","export-01","missing.txt")
    out_path=os.path.join("work_dirs","result","missing_count.csv")
    image_list=load_test_image_from_json(test_json_path)
    seg_missing_dict,v5_missing_dict,v8_missing_dict={},{},{}
    for idx, image in enumerate(image_list):
        seg_missing_dict[image]=0
        v5_missing_dict[image]=0
        v8_missing_dict[image]=0
    seg_missing_dict=load_missing_result(seg_missing_path,seg_missing_dict)
    v5_missing_dict=load_missing_result(v5_missing_path,v5_missing_dict)
    v8_missing_dict=load_missing_result(v8_missing_path,v8_missing_dict)

    result={
        "seg_missing_count":seg_missing_dict,
        "v5_missing_count":v5_missing_dict,
        "v8_missing_count":v8_missing_dict
    }
    result=pd.DataFrame.from_dict(result)
    result.to_csv(out_path)