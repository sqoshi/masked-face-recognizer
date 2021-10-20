import logging

from sklearn.svm import SVC

from predictions.trainers.trainer import Trainer

logger = logging.getLogger(__name__)


class SVMTrainer(Trainer):
    def __init__(self, embeddings):
        model = SVC(C=1.0, kernel="linear", probability=True)
        super().__init__(model, embeddings)

    def train(self):
        logger.info("Training sklearn-svc model with %s 128-D vectors." % len(self._embeddings))
        self._labels = self.label_encoder.fit_transform(self._labels)
        self._model.fit(self._embeddings, self._labels)
        return self._model

    def store_model(self, fn="../svm_model.h5"):
        logger.info("Saving model in %s." % fn)
        super().store_model(fn)
