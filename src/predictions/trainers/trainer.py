import logging
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def read_pickle(path: str) -> Dict[str, np.ndarray]:
    """Loading pickled object from path."""
    with open(path, "rb") as fr:
        return pickle.load(fr)


EmbsDictOrPath = Union[str, Dict[str, List[np.NDArray[Any]]]]


class Trainer(ABC):
    """Model trainer class template."""

    def __init__(self, model, embeddings: EmbsDictOrPath) -> None:
        self._model = model
        self._embeddings = None
        self._labels = None
        self.label_encoder = LabelEncoder()
        self.load_embeddings(embeddings)

    def load_embeddings(self, embeddings) -> None:
        if isinstance(embeddings, str):
            logger.info("Loading embeddings from path %s.", embeddings)
            data = read_pickle(embeddings)
            self._embeddings = data["vectors"]
            self._labels = data["classes"]
        elif isinstance(embeddings, dict):
            logger.info("Loading embeddings from dictionary.")
            self._embeddings = embeddings["vectors"]
            self._labels = embeddings["classes"]
        else:
            raise TypeError("Input must be a dictionary or path to a pickled dict!")

    @abstractmethod
    def train(self):
        pass

    def store_model(self, fp: str = "../model.h5") -> None:
        with open(fp, 'wb') as fw:
            pickle.dump(self._model, fw, protocol=pickle.HIGHEST_PROTOCOL)
