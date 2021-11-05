import logging

from sklearn.svm import SVC

from predictions.trainers.trainer import Trainer
from settings import output

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

    def store_model(self, fn="sklearn_svm_svc_model.pickle"):
        logger.info("Saving model in %s." % (output / fn))
        super().store_model(fn)

    def get_model_details(self):
        return {
            "name": "sklearn.svm.SVC",
            "details": {
                "C": self._model.C,
                "kernel": self._model.kernel,
                "probability": self._model.probability,
                "degree": self._model.degree,
                "gamma": self._model.gamma,
                "coef0": self._model.coef0,
                "tol": self._model.tol,
                "shrinking": self._model.shrinking,
                "class_weight": self._model.class_weight,
                "verbose": self._model.verbose,
                "max_iter": self._model.max_iter,
                "decision_function_shape": self._model.decision_function_shape,
                "break_ties": self._model.break_ties,
                "random_state": self._model.random_state,
            },
        }
