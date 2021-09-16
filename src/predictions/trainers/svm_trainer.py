from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.predictions.trainers.trainer import Trainer


class SVMTrainer(Trainer):
    def __init__(self, embeddings):
        model = SVC(C=1.0, kernel="linear", probability=True)
        super().__init__(model, embeddings)
        self.label_encoder = LabelEncoder()

    def train(self):
        self._labels = self.label_encoder.fit_transform(self._labels)
        self._model.fit(self._embeddings, self._labels)
        return self._model

    def store_model(self, fn="../svm_model.h5"):
        super().store_model(fn)
