import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

from predictions.face_recognizer import ModelType
from settings import output

logger = logging.getLogger(__name__)


def read_pickle(path: str) -> Dict[str, NDArray[Any]]:
    """Loading pickled object from path."""
    with open(path, "rb") as fr:
        return pickle.load(fr)


EmbsDictOrPath = Union[str, Dict[str, List[NDArray[Any]]]]


class Trainer(ABC):
    """Model trainer class template."""

    def __init__(self, model: ModelType, embeddings: EmbsDictOrPath) -> None:
        self._model = model
        self._embeddings: Optional[List[Any]] = None
        self._labels: Optional[List[Any]] = None
        self.label_encoder = LabelEncoder()
        self.load_embeddings(embeddings)

    def load_embeddings(self, embeddings: EmbsDictOrPath) -> None:
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
    def train(self) -> None:
        pass

    def store_model(self, fn: str = "model.pickle") -> None:
        """Saves model in directory as pickle."""
        with open(output / fn, "wb") as fw:
            pickle.dump(self._model, fw, protocol=pickle.HIGHEST_PROTOCOL)
