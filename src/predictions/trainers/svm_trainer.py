import logging
import time

from sklearn.svm import SVC

from predictions.trainers.trainer import Trainer
from settings import output

logger = logging.getLogger(__name__)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        logger.info(f"Execution time of {func.__name__} have taken {(time.time() - start) / 60} minutes.")

    return wrapper


class SVMTrainer(Trainer):
    def __init__(self, embeddings):
        model = SVC(C=1.0, kernel="linear", probability=True)
        super().__init__(model, embeddings)

    @timer
    def train(self):
        logger.info(
            "Training sklearn-svc model with %s 128-D vectors." % len(self._embeddings)
        )
        self._labels = self.label_encoder.fit_transform(self._labels)
        self._model.fit(self._embeddings, self._labels)
        return self._model

    def store_model(self, fn="svm_model.h5"):
        logger.info("Saving model in %s." % (output / fn))
        super().store_model(fn)
