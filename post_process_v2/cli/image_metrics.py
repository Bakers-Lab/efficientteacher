from typing import Tuple, Dict, List
import pandas as pd

TOTAL_COUNT = 'total_count'
GT_PASS_COUNT = 'gt_pass_count'
GT_NG_COUNT = 'gt_ng_count'
ACC_RATE = 'acc_rate'
ACC_COUNT = 'acc_count'
OVERKILL_RATE = 'overkill_rate'
OVERKILL_COUNT = 'overkill_count'
WRONG_RATE = 'wrong_rate'
WRONG_COUNT = 'wrong_count'
MISS_RATE = 'miss_rate'
MISS_COUNT = 'miss_count'

METRIC_RATE_COL_NAMES = [ACC_RATE, OVERKILL_RATE, MISS_RATE]
METRIC_COUNT_COL_NAMES = [TOTAL_COUNT, GT_PASS_COUNT, GT_NG_COUNT, ACC_COUNT, OVERKILL_COUNT, MISS_COUNT]
METRIC_COL_NAMES = [TOTAL_COUNT, GT_PASS_COUNT, GT_NG_COUNT, ACC_RATE, ACC_COUNT, MISS_RATE, MISS_COUNT, OVERKILL_RATE,
                    OVERKILL_COUNT]

GT_COLUMN_NAMES_RAW = ["CenterX", "CenterY", "Length", "Width", "ImageID", "LableID", "Area", "BoxID"]
GT_COLUMN_NAMES_OLD = ["CenterX", "CenterY", "Length", "Width", "ImageID", "LableID", "BoxID"]
GT_COLUMN_NAMES_NEW = ['center_x', 'center_y', 'length', 'width', 'image_id', 'label_id', 'box_id']

PREDS_COLUMN_NAMES_OLD = ["CenterX", "CenterY", "Length", "Width", "Confidence", "ImageID", "LableID", "Area", "BoxID"]
PREDS_COLUMN_NAMES_NEW = [
    'center_x', 'center_y', 'length', 'width', 'confidence', 'image_id', 'label_id', 'area', 'box_id',
]

'''
protected
'''


def _get_img2label(df: pd.DataFrame, img_ids: List[int]) -> Dict[int, List[int]]:
    img2label = {img_id: None for img_id in img_ids}
    for img_id in img2label:
        img2label[img_id] = df[df['image_id'] == img_id]['label_id'].unique().tolist()
    return img2label


def _get_pass_ng_imgs(img2label: Dict, pass_id: int) -> Tuple[List[int], List[int]]:
    pass_imgs, ng_imgs = [], []
    for img_id in img2label:
        if img2label[img_id] == [pass_id] or len(img2label[img_id]) == 0:
            pass_imgs.append(img_id)
        else:
            ng_imgs.append(img_id)
    return pass_imgs, ng_imgs


'''
public
'''


def load_data_csv(pred_filepath: str, gt_filepath: str):
    # GT 数据
    gt_df = pd.read_csv(gt_filepath, names=GT_COLUMN_NAMES_RAW, header=None)
    gt_df = gt_df[GT_COLUMN_NAMES_OLD]
    gt_df.columns = GT_COLUMN_NAMES_NEW
    # TPMBDBDL 数据
    pred_df = pd.read_csv(pred_filepath, header='infer')
    pred_df = pred_df[PREDS_COLUMN_NAMES_OLD]
    pred_df.columns = PREDS_COLUMN_NAMES_NEW
    return pred_df, gt_df


def get_img_metric(
        pred_df: pd.DataFrame, gt_df: pd.DataFrame,
        pass_label_id: int,
) -> Tuple[Dict, Dict]:
    img_ids = gt_df['image_id'].unique()
    pred_img2label = _get_img2label(pred_df, img_ids)
    gt_img2label = _get_img2label(gt_df, img_ids)
    pred_pass_imgs, pred_ng_imgs = _get_pass_ng_imgs(pred_img2label, pass_label_id)
    gt_pass_imgs, gt_ng_imgs = _get_pass_ng_imgs(gt_img2label, pass_label_id)
    acc_count = len(set(pred_pass_imgs).intersection(set(gt_pass_imgs))) + len(
        set(pred_ng_imgs).intersection((set(gt_ng_imgs))))
    miss_image_set = set(pred_pass_imgs).intersection(set(gt_ng_imgs))
    overkill_image_set = set(pred_ng_imgs).intersection(set(gt_pass_imgs))
    metric = {
        TOTAL_COUNT: len(img_ids),
        GT_PASS_COUNT: len(gt_pass_imgs),
        GT_NG_COUNT: len(gt_ng_imgs),
        ACC_RATE: acc_count / (len(img_ids) or 1),
        ACC_COUNT: acc_count,
        MISS_RATE: len(miss_image_set) / (len(img_ids) or 1),
        MISS_COUNT: len(miss_image_set),
        OVERKILL_RATE: len(overkill_image_set) / (len(img_ids) or 1),
        OVERKILL_COUNT: len(overkill_image_set)
    }
    image_ids = {
        "missing": [int(x) for x in miss_image_set],
        "overkill": [int(x) for x in overkill_image_set],
    }
    return metric, image_ids


def image_metrics_to_csv(metric: Dict, save_path: str):
    raw_report = pd.DataFrame([metric])
    rate_report = raw_report[METRIC_RATE_COL_NAMES]
    count_report = raw_report[METRIC_COUNT_COL_NAMES]
    rate_report = (100. * rate_report).round(2).astype(str) + '%'
    report = pd.concat([rate_report, count_report], axis=1)
    report = report[METRIC_COL_NAMES]
    report.to_csv(save_path, index=None)
