import json
import logging
import os
import time
from collections import namedtuple
from typing import Optional, Union

DatasetModifications = namedtuple("DatasetModifications", "mask_ratio inplace")

logger = logging.getLogger(__file__)


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
            return DatasetModifications(0.0, True)
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
                "train": {
                    "mask_ratio": self.modifications.train.mask_ratio,
                    "inplace": self.modifications.train.inplace,
                },
                "test": {
                    "mask_ratio": self.modifications.test.mask_ratio,
                    "inplace": self.modifications.test.inplace,
                },
            },
        }

    def to_json(self, fp: str) -> None:
        with open(fp, "w+") as fw:
            json.dump(self.as_dict(), fw)
