import json
import logging
import time
from collections import namedtuple
from typing import Optional, Union

from settings import output

DatasetModifications = namedtuple("DatasetModifications", "mask_ratio inplace skip_unknown")

logger = logging.getLogger(__file__)


class AnalysisConfig:
    def __init__(
            self,
            dataset_path: str,
            split_ratio: float,
            personal_images_quantity: int,
            is_piq_max: bool = True,
            identities_limit: Optional[int] = None,
            test_set_modifications: Optional[Union[tuple, DatasetModifications]] = None,
            train_set_modifications: Optional[Union[tuple, DatasetModifications]] = None,
    ):
        self.dataset_path: str = dataset_path
        self.split_ratio: float = split_ratio
        self.personal_images_quantity: int = personal_images_quantity
        self.is_piq_max: bool = is_piq_max
        self.identities_limit = identities_limit
        self.modifications = namedtuple("modifications", "test train")
        self.modifications.test = self.create_modifications(test_set_modifications)
        self.modifications.train = self.create_modifications(train_set_modifications)

    @staticmethod
    def create_modifications(
            modifications: Union[tuple, DatasetModifications]
    ) -> Optional[DatasetModifications]:
        if isinstance(modifications, tuple):
            return DatasetModifications(*modifications)
        elif isinstance(modifications, DatasetModifications):
            return modifications
        elif modifications is None:
            return DatasetModifications(0.0, True, False)
        raise TypeError("Tuple and DatasetModifications is only accepted.")

    # def values(self):
    #     return (
    #         self.dataset_path,
    #         self.split_ratio,
    #         self.modifications.test,
    #         self.modifications.train,
    #         self.personal_images_quantity,
    #     )
    #
    # def as_list(self):
    #     return [*self.values()]

    def as_dict(self):
        return {
            "dataset_path": self.dataset_path,
            "personal_images_quantity": self.personal_images_quantity,
            "equal_piqs": self.is_piq_max,
            "identities_limit": self.identities_limit,
            "split_ratio": self.split_ratio,
            "modifications": {
                {
                    "train": {
                        "mask_ratio": self.modifications.train.mask_ratio,
                        "inplace": self.modifications.train.inplace,
                        "skip_unknown": self.modifications.train.skip_unknown,
                    },
                    "test": {
                        "mask_ratio": self.modifications.test.mask_ratio,
                        "inplace": self.modifications.test.inplace,
                        "skip_unknown": self.modifications.test.skip_unknown,
                    }
                    # "test": dict(self.modifications.test._asdict()),
                }
            },
        }

    def to_json(self, fn: Optional[str] = None) -> None:
        if fn is None:
            fn = f"analysis_config_{time.time()}.json"
        with open(output / fn, "w+") as fw:
            json.dump(self.as_dict(), fw)
