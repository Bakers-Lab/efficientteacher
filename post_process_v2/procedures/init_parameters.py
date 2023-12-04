from typing import Union, Dict, List


class Threshold:
    DATA_KEYS = ("label_id", "label_name", "min_confidence")

    def __init__(
            self, label_id: int, label_name: str,
            min_confidence: Union[float, Dict[int, float]],
    ):
        self.label_id = label_id
        self.label_name = label_name
        # 如果 min_confidence 为 float 类型, 表示各 LabelIndex 使用统一阈值
        # 如果 min_confidence 为 dict 类型, 表示不同 LabelIndex 使用各自的阈值
        self.min_confidence = self.__init_min_confidence(min_confidence)

    def __init_min_confidence(self, min_confidence: Union[float, Dict[int, float]]):
        result = min_confidence
        if isinstance(result, dict):
            result = {int(x): y for x, y in result.items()}
        return result

    def is_row_ok(self, row: dict) -> bool:
        if row["LableID"] != self.label_id:
            return False
        if isinstance(self.min_confidence, float):
            # 各 LabelIndex 使用统一阈值
            _row_min_confi = self.min_confidence
        else:
            # 各 LabelIndex 使用不同阈值
            # 当某行的 LabelIndex 未设置阈值时，确保返回状态为 False
            _row_min_confi = self.min_confidence.get(row["LableIndex"], 1 + row["Confidence"])
        return row["Confidence"] > _row_min_confi

    def to_dict(self) -> dict:
        return {x: getattr(self, x) for x in self.DATA_KEYS}

    def copy(self):
        data = {x: y.copy() if isinstance(y, dict) else y for x, y in self.to_dict().items()}
        return Threshold(**data)


def init_parameters(id_name_dict: dict) -> Dict[str, Threshold]:
    return {
        _name: Threshold(_id, _name, min_confidence=0.0)
        for _id, _name in id_name_dict.items()
    }


'''
example
'''


def parameters_example() -> List[Dict[str, Threshold]]:
    result = (
        # config: OP -> P
        {
            "DZ_CM": {"label_id": 0, "min_confidence": 0.0022},
            "DZ_FM": {"label_id": 1, "min_confidence": 0.0009},
            "DZ_MC": {"label_id": 2, "min_confidence": 0.0012},
            "DZ_PASS": {"label_id": 3, "min_confidence": 0.0012},
        },
        # config: PMBDB -> PMBDBDL
        {
            "DZ_CM": {"label_id": 0, "min_confidence": {1: 0.027800000000000002, 2: 0.008700000000000001, 3: 0.0022}},
            "DZ_FM": {"label_id": 1, "min_confidence": {1: 0.0009, 2: 0.0039000000000000003, 3: 0.006}},
            "DZ_MC": {"label_id": 2, "min_confidence": {1: 0.0012, 2: 0.0063}},
            "DZ_PASS": {"label_id": 3, "min_confidence": {1: 0.0012, 2: 0.0019, 3: 0.0021000000000000003}},
        }
    )
    result = [
        {_name: Threshold(label_name=_name, **_info) for _name, _info in line.items()}
        for line in result
    ]
    return result
