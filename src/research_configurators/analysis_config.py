import json
import logging
import os
import time
from collections import namedtuple
from typing import Optional, Union, Dict

DatasetModifications = namedtuple("DatasetModifications", "mask_ratio inplace mask")

logger = logging.getLogger(__file__)


def namedtuple_asdict(namedtuple_object) -> Dict[str, str]:
    """Converts namedtuple to a strings dictionary."""
    res = {}
    for k, v in zip(namedtuple_object._fields, namedtuple_object):
        res[k] = str(v)
    return res


class AnalysisConfig:
    def __init__(
            self,
            dataset_path: str,
            split_ratio: float,
            personal_images_quantity: int = 1,  # default value selects
            equal_piqs: bool = False,
            skip_unknown: bool = False,
            identities_limit: Optional[int] = None,
            test_set_modifications: Optional[Union[tuple, DatasetModifications]] = None,
            train_set_modifications: Optional[Union[tuple, DatasetModifications]] = None,
            name: Optional[str] = None,
    ):
        self.dataset_path: str = dataset_path
        self.split_ratio: float = split_ratio
        self.personal_images_quantity: int = personal_images_quantity
        self.equal_piqs: bool = equal_piqs
        self.identities_limit = identities_limit
        self.skip_unknown: bool = skip_unknown
        self.modifications = namedtuple("modifications", "test train")
        self.modifications.test = self.create_modifications(test_set_modifications)
        self.modifications.train = self.create_modifications(train_set_modifications)
        self.name = self.set_name(name)

    def get_dataset_name(self):
        return self.dataset_path.split(os.path.sep)[-1]

    def set_name(self, name) -> str:
        if name is not None:
            return name
        return f"{self.get_dataset_name()}_{int(time.time())}"

    @staticmethod
    def create_modifications(
            modifications: Union[tuple, DatasetModifications]
    ) -> Optional[DatasetModifications]:
        if isinstance(modifications, tuple):
            return DatasetModifications(*modifications)
        elif isinstance(modifications, DatasetModifications):
            return modifications
        elif modifications is None:
            return DatasetModifications(0.0, True, 1)
        raise TypeError("Tuple and DatasetModifications is only accepted.")

    def as_dict(self):
        return {
            "dataset_path": self.dataset_path,
            "personal_images_quantity": self.personal_images_quantity,
            "equal_piqs": self.equal_piqs,
            "identities_limit": self.identities_limit,
            "split_ratio": self.split_ratio,
            "skip_unknown": self.skip_unknown,
            "modifications": {
                "train": namedtuple_asdict(self.modifications.train),
                "test": namedtuple_asdict(self.modifications.test),
            },
        }

    def to_json(self, fp: str) -> None:
        with open(fp, "w+") as fw:
            json.dump(self.as_dict(), fw)
