import json
import logging
import os
import time
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

DatasetModifications = namedtuple("DatasetModifications", "mask_ratio inplace mask")

logger = logging.getLogger(__file__)


def namedtuple_asdict(namedtuple_object: namedtuple) -> Dict[str, str]:  # type:ignore
    """Converts namedtuple to a strings dictionary."""
    res = {}
    for k, v in zip(namedtuple_object._fields, namedtuple_object):  # type:ignore
        res[k] = str(v)
    return res


class Configuration:  # pylint:disable=too-many-instance-attributes
    """Contains modifications that should be used in dataset reading,
     learning and also model structure setup.
    """

    def __init__(  # pylint:disable=too-many-arguments
        self,
        dataset_path: str,
        split_ratio: float,
        personal_images_quantity: int = 1,  # default value selects
        equal_piqs: bool = False,
        skip_unknown: bool = False,
        identities_limit: Optional[int] = None,
        landmarks_detection: bool = True,
        test_set_modifications: Optional[
            Union[Tuple[Any, Any], DatasetModifications]
        ] = None,
        train_set_modifications: Optional[
            Union[Tuple[Any, Any], DatasetModifications]
        ] = None,
        name: Optional[str] = None,
        svm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dataset_path: str = dataset_path
        self.split_ratio: float = split_ratio
        self.personal_images_quantity: int = personal_images_quantity
        self.equal_piqs: bool = equal_piqs
        self.identities_limit = identities_limit
        self.skip_unknown: bool = skip_unknown
        self.landmarks_detection: bool = landmarks_detection
        self.modifications = namedtuple("modifications", "test train")  # type:ignore
        self.modifications.test = self.create_modifications(  # type:ignore
            test_set_modifications  # type:ignore
        )
        self.modifications.train = self.create_modifications(  # type:ignore
            train_set_modifications  # type:ignore
        )
        self.name = self.set_name(name)
        self.svm_config = (
            svm_config
            if svm_config is not None
            else {
                "C": 1.0,
                "kernel": "poly",
                "degree": 5,
                "probability": True,
                "random_state": True,
            }
        )

    def get_dataset_name(self) -> str:
        """Gets dataset name from path to dataset."""
        return self.dataset_path.split(os.path.sep)[-1]

    def set_name(self, name: Optional[str]) -> str:
        """Sets name to dataset_name_current_time_integer"""
        if name is not None:
            return name
        return f"{self.get_dataset_name()}_{int(time.time())}"

    @staticmethod
    def create_modifications(
        modifications: Union[Tuple[Any, Any], DatasetModifications]
    ) -> Optional[DatasetModifications]:
        """Unifies modification passed in different dat structures to DatasetModifications obj."""
        if isinstance(modifications, tuple):
            return DatasetModifications(*modifications)
        if isinstance(modifications, DatasetModifications):
            return modifications
        if modifications is None:
            return DatasetModifications(0.0, True, 1)
        raise TypeError("Tuple and DatasetModifications is only accepted.")

    def as_dict(self) -> Dict[str, Any]:
        """Dictionary with configuration"""
        return {
            "dataset_path": self.dataset_path,
            "personal_images_quantity": self.personal_images_quantity,
            "equal_piqs": self.equal_piqs,
            "identities_limit": self.identities_limit,
            "split_ratio": self.split_ratio,
            "skip_unknown": self.skip_unknown,
            "landmarks_detection": self.landmarks_detection,
            "modifications": {
                "train": namedtuple_asdict(self.modifications.train),
                "test": namedtuple_asdict(self.modifications.test),
            },
        }

    def to_json(self, fp: Union[Path, str]) -> None:
        """Save configuration as json."""
        with open(fp, "w+") as fw:
            json.dump(self.as_dict(), fw)
