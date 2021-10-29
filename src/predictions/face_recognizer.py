import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from termcolor import colored, cprint

from predictions.embedder import Embedder
from predictions.face_detector import FaceDetector
from settings import output

logger = logging.getLogger(__name__)

ModelType = Union[SVC]


class FaceRecognizer(FaceDetector, Embedder):
    """Class responsible for recognize classes on images."""

    def __init__(self, model, label_encoder: LabelEncoder) -> None:
        FaceDetector.__init__(self)
        Embedder.__init__(self)
        self._model = model
        self._standard_width = 600
        self.label_encoder = label_encoder
        self.statistics = {
            "stats": {x: 0 for x in ("perfect", "top5", "fail")},
            "accuracy": {x: 0 for x in ("perfect", "top5")},
            "personal_stats": {},
        }

    @property
    def stats(self):
        return self.statistics["stats"]

    def personal_stats(self, identity):
        if identity not in self.statistics["personal_stats"].keys():
            self.statistics["personal_stats"][identity] = {
                x: 0 for x in ("perfect", "top5", "fail")
            }
        return self.statistics["personal_stats"][identity]

    @property
    def accuracy(self):
        return self.statistics["accuracy"]

    def update_stats(
        self, predictions: np.ndarray, correct_identity: str, stats: Dict[str, int]
    ) -> None:
        """Analyzes predictions and update statistics."""
        arr = np.array(predictions)
        top5_indexes = arr.argsort()[-5:][::-1]
        classes = [self.label_encoder.classes_[j] for j in top5_indexes]
        probs = [predictions[idx] for idx in top5_indexes]

        if correct_identity in classes:
            if probs[classes.index(correct_identity)] == max(probs):
                m = colored("PERFECT", "green")
                key = "perfect"
            else:
                m = colored("TOP5", "yellow")
                key = "top5"
        else:
            m = colored("WRONG", "red")
            key = "fail"

        print(f"{correct_identity} - {m} match.")
        stats[key] += 1
        self.personal_stats(correct_identity)[key] += 1

    def print_stats(self, stats) -> None:
        """Prints simple statistics."""
        cprint(f"{stats['perfect']} perfects.", "green")
        cprint(f"{stats['top5']} top5.", "yellow")
        cprint(f"{stats['fail']} fails.", "red")
        cprint(f"Model accuracy [perfect]: {self.accuracy['perfect']}", "green")
        cprint(f"Model accuracy [top5+perfect]: {self.accuracy['top5']}", "green")

    def compute_accuracy(self, stats) -> None:
        """Compute overall model accuracy."""
        tests_number = sum(stats.values())
        self.accuracy["top5"] = round((stats["perfect"] + stats["top5"]) / tests_number * 100, 3)
        self.accuracy["perfect"] = round((stats["perfect"]) / tests_number * 100, 3)

    def save_stats(self, stats_fp: str = "statistics.csv", append_accuracy=False) -> None:
        """Saves statistic in csv file."""
        df = pd.DataFrame.from_dict(self.statistics["personal_stats"]).T
        if append_accuracy:
            sm = df["fail"] + df["perfect"] + df["top5"]
            df["perfect_accuracy"] = round(df["perfect"] / sm * 100, 3)
            df["top5_accuracy"] = round((df["perfect"] + df["top5"]) / sm * 100, 3)
        df.to_csv(output / stats_fp)

    def recognize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classifies identities on images and collects statistics."""
        for i, (vec, img) in enumerate(self.vector_generator(df, self.vector)):
            logger.info(f"Recognizing (%s/%s) ...", i, len(df.index))

            preds = self._model.predict_proba(vec)[0]
            self.update_stats(preds, img.identity, self.stats)

        self.compute_accuracy(self.stats)
        self.print_stats(self.stats)
        # self.save_stats()
        return pd.DataFrame.from_dict(self.statistics["personal_stats"]).T
