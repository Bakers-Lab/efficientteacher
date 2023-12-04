from post_process_v2.base.helpers import method_run_once_limit

from typing import Tuple, Dict, List
import math


class BBox:
    KEY_MAP_DICT = {
        "image_id": "ImageID", "confidence": "Confidence", "label_id": "LableID", "center_x": "CenterX",
        "center_y": "CenterY", "length": "Length", "width": "Width", "box_id": "BoxID",
    }

    def __init__(
            self, image_id: int, confidence: float, label_id: int,
            center_x: float, center_y: float, length: float, width: float,
            box_id: int = None, area: float = None,
    ):
        self.image_id = image_id
        self.confidence = confidence
        self.label_id = label_id
        #
        self.center_x = center_x
        self.center_y = center_y
        self.length = length
        self.width = width
        #
        self.box_id = box_id
        self.merged = False
        self.merge_history = set()
        self._area = area

    '''
    property & static
    '''

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self.length * self.width
        return self._area

    @staticmethod
    def load_from_dict(data: dict):
        new_data = {x: data[y] for x, y in BBox.KEY_MAP_DICT.items()}
        # 注意:
        # 对于 OP 原始数据，Length, Width, Area 均只保留了 4 位小数，这会导致 Length * Width 与 Area 之间的偏差很大（大到影响最终的 ratio 的程度）
        # 为避免精度差异造成的错误，如果从 OP 原始数据构造 BBox 对象，应使用 OP 数据原始的 Area，而不是每次都根据 length 和 width 重新计算
        new_data["area"] = data.get("Area", None)
        return BBox(**new_data)

    '''
    public
    '''

    def to_dict(self, override_data: dict = None) -> dict:
        result = {y: getattr(self, x) for x, y in self.KEY_MAP_DICT.items()}
        result["Area"] = self.area
        result.update(override_data or dict())
        return result

    def update_merge_history(self, *boxes):
        for _box in boxes:
            if _box.box_id is not None:
                self.merge_history.add(_box.box_id)
            self.merge_history.update(_box.merge_history)

    def copy(self):
        result = self.load_from_dict(self.to_dict())
        result.merge_history = self.merge_history.copy()
        return result


def calc_merged_bounding(box1: BBox, box2: BBox, outer_mode=True) -> Tuple[float, float, float, float]:
    right_top_func = max if outer_mode else min
    left_bottom_func = min if outer_mode else max
    left = left_bottom_func(box1.center_x - box1.length / 2, box2.center_x - box2.length / 2)
    right = right_top_func(box1.center_x + box1.length / 2, box2.center_x + box2.length / 2)
    top = right_top_func(box1.center_y + box1.width / 2, box2.center_y + box2.width / 2)
    bottom = left_bottom_func(box1.center_y - box1.width / 2, box2.center_y - box2.width / 2)
    return left, right, top, bottom


def merge_boxes_with_distance(box1: BBox, box2: BBox, with_history=False) -> Tuple[float, BBox]:
    if box1.image_id != box2.image_id:
        raise ValueError("不允许合并来自不同 image 的检测框")
    # 基础坐标
    left, right, top, bottom = calc_merged_bounding(box1, box2, outer_mode=True)
    distance = math.sqrt((box1.center_x - box2.center_x) ** 2 + (box1.center_y - box2.center_y) ** 2)
    # 构造 box 数据
    result = BBox(
        image_id=box1.image_id, box_id=None,
        center_x=(left + right) / 2, center_y=(top + bottom) / 2, length=right - left, width=top - bottom,
        # 保留高置信度 box 的特征
        confidence=max(box1.confidence, box2.confidence),
        label_id=box1.label_id if box1.confidence > box2.confidence else box2.label_id,
    )
    # 记录合并历史
    if with_history:
        result.update_merge_history(box1, box2)
    return distance, result


def merge_boxes(box1: BBox, box2: BBox, with_history=False) -> BBox:
    _, result = merge_boxes_with_distance(box1, box2, with_history)
    return result


class BoundBoxMerger:
    def __init__(
            self, image_boxes: List[BBox],
            area_threshold: float, length_threshold: float, iou_threshold: float
    ):
        self.image_boxes = image_boxes
        self.image_box_dict = {x.box_id: x for x in self.image_boxes}
        # 阈值
        self.area_threshold = area_threshold
        self.length_threshold = length_threshold
        self.iou_threshold = iou_threshold
        # values
        self._max_box_id = 1 + max([x.box_id for x in self.image_boxes])

    '''
    private
    '''

    def __is_iou_ok(self, box1: BBox, box2: BBox) -> bool:
        left, right, top, bottom = calc_merged_bounding(box1, box2, outer_mode=False)
        # 计算 iou
        is_inter_sect = abs(box1.center_x - box2.center_x) <= (box1.length + box2.length) / 2 \
                        or abs(box1.center_y - box2.center_y) <= (box1.width + box2.width) / 2
        min_area = min(box1.area, box2.area)
        iou = 0 if min_area <= 0 else (right - left) * (top - bottom) / min_area
        return is_inter_sect and iou > self.iou_threshold

    def __bbox_to_pmb_rows(self, box: BBox) -> List[Dict]:
        max_confidence_dict = dict()
        # 计算各类别的置信度
        for _id in box.merge_history or [box.box_id]:
            _row = box if _id == box.box_id else self.image_box_dict.get(_id, None)
            # 确实存在合并来源的 box 不存在的情形
            if not _row:
                continue
            _label_id = _row.label_id
            max_confidence_dict[_label_id] = max(_row.confidence, max_confidence_dict.get(_label_id, -1))
        # 构建行数据
        result = []
        for _label, _confidence in max_confidence_dict.items():
            result.append(box.to_dict({"LableID": _label, "Confidence": _confidence}))
        # 按照 LabelID 排序
        result.sort(key=lambda x: x["LableID"])
        return result

    '''
    protected
    '''

    def _merge_boxes_by_length_area(self, box_list: List[BBox]) -> List[BBox]:
        for _row in box_list:
            _row.merged = False
        # 两两合并: box1 + box2 -> box3
        result = []
        while True:
            box_list = [x for x in box_list if not x.merged]
            box_list.sort(key=lambda x: x.confidence, reverse=True)
            if not box_list:
                break
            # 置信度最高的框作为 box1
            box1 = box_list[0]
            box1.merged = True
            # 选择与 box1 距离最近的作为 box2
            dist_list = []
            for box2 in box_list[1:]:
                distance, box3 = merge_boxes_with_distance(box1, box2)
                if box3.length >= self.length_threshold \
                        or box3.width >= self.length_threshold \
                        or box3.area >= self.area_threshold:
                    continue
                dist_list.append((distance, box2, box3))
            # 此 box1 无可用的合并
            if not dist_list:
                result.append(box1)
                continue
            # 取 box3
            dist_list.sort(key=lambda x: x[0])
            _, box2, box3 = dist_list[0]
            box2.merged = True
            box3.box_id = self._max_box_id
            self._max_box_id += 1
            # 将合并得到的 box3 放回 box 集合
            box_list.append(box3)
            self.image_boxes.append(box3)
            self.image_box_dict[box3.box_id] = box3
            # 记录合并历史
            box3.update_merge_history(box1, box2)
        return result

    def _merge_boxes_by_iou(self, box_list: List[BBox]) -> List[Dict]:
        for _row in box_list:
            _row.merged = False
        # 多个合并为一个: box1 + (box2_1 + box2_2 ...) -> box_3
        result = []
        while True:
            box_list = [x for x in box_list if not x.merged]
            if not box_list:
                break
            # 任选一个作为 box1
            box1 = box_list[0]
            box1.merged = True
            # 根据 IoU 选出多个 box2。将这些 box2 与 box1 合并为 box3
            box3 = box1.copy()
            merge_count = 0
            for box2 in box_list[1:]:
                if not self.__is_iou_ok(box1, box2):
                    continue
                box2.merged = True
                box3 = merge_boxes(box3, box2, with_history=True)
                merge_count += 1
            # 设置 BoxID: 当 merge_count == 0, 说明不是新的 box, 仍使用旧的 box1.box_id
            if not merge_count:
                box3.box_id = box1.box_id
            else:
                box3.box_id = self._max_box_id
                self._max_box_id += 1
            # 基于 box3 的 from_box，计算 box3 各类 label 的置信度
            result.extend(self.__bbox_to_pmb_rows(box3))
        return result

    '''
    public
    '''

    @method_run_once_limit
    def run(self) -> List[Dict]:
        result = self._merge_boxes_by_length_area(self.image_boxes)
        result = self._merge_boxes_by_iou(result)
        return result
