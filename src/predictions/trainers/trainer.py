import pickle
from abc import ABC, abstractmethod


def read_pickle(path: str):
    with open(path, "rb") as fr:
        return pickle.load(fr)


class Trainer(ABC):
    def __init__(self, model, embeddings):
        self._model = model
        self._embeddings = None
        self._labels = None
        self.read_embeddings(embeddings)

    def read_embeddings(self, embeddings):
        if isinstance(embeddings, str):
            data = read_pickle(embeddings)
            self._embeddings = data["vectors"]
            self._labels = data["classes"]
        elif isinstance(embeddings, dict):
            self._embeddings = embeddings["vectors"]
            self._labels = embeddings["classes"]
        else:
            raise TypeError("Input must be a dictionary or path to a pickled dict!")

    @abstractmethod
    def train(self):
        pass

    def store_model(self, fp="../model.h5"):
        with open(fp, 'wb') as fw:
            pickle.dump(self._model, fw, protocol=pickle.HIGHEST_PROTOCOL)
