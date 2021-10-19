import logging
from collections import namedtuple

import coloredlogs
import pandas as pd

from preanalysis.dataset_config import DatasetConfigBuilder
from preanalysis.dataset_modificator import DatasetModifier
from preanalysis.reader import DatasetReader
from predictions.extractor import FaceExtractor
from predictions.face_recognizer import FaceRecognizer
from predictions.trainers.svm_trainer import SVMTrainer

logging.basicConfig(filename="masked_face_recognizer.log")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

DatasetModifications = namedtuple("DatasetModifications", "ratio inplace")


class Analyzer:
    def __init__(self):
        self._dsc_builder = DatasetConfigBuilder()
        self._dataset_reader = DatasetReader()
        self._dataset_modifier = DatasetModifier()
        self._face_extractor = FaceExtractor()

    def read_dataset(self, path: str) -> pd.DataFrame:
        return self._dataset_reader.read(self._dsc_builder.build(path))

    def reset(self) -> None:
        self._face_extractor.reset()

    def run(
            self,
            dataset_path: str,
            dataset_split_ratio: float,
            train_set_modifications: DatasetModifications,
            test_set_modifications: DatasetModifications,
    ):
        logger.info("Program started.")

        logger.info("1. Dataset reading stage.")
        dataset = self.read_dataset(dataset_path)
        train_set, test_set = self._dataset_reader.split_dataset(
            dataset, ratio=dataset_split_ratio
        )

        logger.info("2. Extracting embeddings stage.")
        embs = self._face_extractor.extract(
            self._dataset_modifier.modify(train_set, *train_set_modifications)
        )
        # fd.save()
        # embs = "../face_vectors.pickle"

        logger.info("3. Model training stage")
        t = SVMTrainer(embs)
        m = t.train()
        label_coder = t.label_encoder
        # t.store_model()

        logger.info("4. Face recognition stage.")
        fr = FaceRecognizer(m, label_coder)
        fr.recognize(self._dataset_modifier.modify(test_set, *test_set_modifications))

        logger.info("Program ended.")


if __name__ == "__main__":
    standard_path = "/home/piotr/Documents/bsc-thesis/datasets/original"
    analyzer = Analyzer()
    analyzer.run(
        standard_path,
        dataset_split_ratio=0.8,
        train_set_modifications=DatasetModifications(0.0, False),
        test_set_modifications=DatasetModifications(0.0, True),
    )
    analyzer.reset()
