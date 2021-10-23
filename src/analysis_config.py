from collections import namedtuple
from typing import Union, Optional

# original = "/home/piotr/Documents/bsc-thesis/datasets/test"
# original = "/home/piotr/Documents/bsc-thesis/datasets/test"
# original = "/home/piotr/Documents/bsc-thesis/datasets/original"
celeba = "/home/piotr/Documents/bsc-thesis/datasets/celeba"

DatasetModifications = namedtuple(
    "DatasetModifications", "mask_ratio inplace skip_unknown"
)


class AnalysisConfig:
    def __init__(
            self,
            dataset: str,
            split_ratio: float,
            personal_images_quantity: int = 1,
            test_set_modifications: Optional[Union[tuple, DatasetModifications]] = None,
            train_set_modifications: Optional[Union[tuple, DatasetModifications]] = None,
    ):
        self.dataset: str = dataset
        self.split_ratio: float = split_ratio
        self.personal_images_quantity: int = personal_images_quantity
        self.modifications = namedtuple("modifications", "test train")
        self.modifications.test = self.create_modifications(test_set_modifications)
        self.modifications.train = self.create_modifications(train_set_modifications)

    @staticmethod
    def create_modifications(
            modifications: Union[tuple, DatasetModifications]
    ) -> Optional[DatasetModifications]:
        if isinstance(modifications, tuple):
            return DatasetModifications(*modifications)
        elif isinstance(modifications, DatasetModifications) or modifications is None:
            return modifications
        raise TypeError("Tuple and DatasetModifications is only accepted.")

    def values(self):
        return (
            self.dataset,
            self.split_ratio,
            self.modifications.test,
            self.modifications.train,
            self.personal_images_quantity,
        )

    def as_list(self):
        return [*self.values()]

    def as_dict(self):
        return {
            "dataset": self.dataset,
            "personal_images_quantity": self.personal_images_quantity,
            "split_ratio": self.split_ratio,
            "modifications": {
                {
                    "train": dict(self.modifications.train._asdict()),
                    "test": dict(self.modifications.test._asdict()),
                }
            },
        }


configs = [
    # AnalysisConfig(original, 0.8, 1, None, None),
    AnalysisConfig(celeba, 0.8, 25, None, None),
    # (standard_path, 0.8, None, DatasetModifications(1.0, True, False)),
    # (standard_path, 0.8, DatasetModifications(0.2, True, False), DatasetModifications(1.0, True, False)),
    # (standard_path, 0.8, DatasetModifications(0.2, False), DatasetModifications(1.0, True)),
    # (standard_path, 0.8, DatasetModifications(0.5, True), DatasetModifications(1.0, True)),
    # (standard_path, 0.8, DatasetModifications(0.5, False), DatasetModifications(1.0, True)),
    # (standard_path, 0.8, DatasetModifications(1.0, True), DatasetModifications(1.0, True)),
    # (standard_path, 0.8, DatasetModifications(1.0, False), DatasetModifications(1.0, True)),
]
