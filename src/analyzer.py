import csv
import logging
import os
import time
from typing import Optional

import coloredlogs
import pandas as pd

from preanalysis.dataset_utils.dataset_configuration.dataset_config_builder import DatasetConfigBuilder
from preanalysis.dataset_utils.dataset_modification.dataset_modificator import DatasetModifier
from preanalysis.dataset_utils.dataset_reading.dataset_reader import DatasetReader
from predictions.extractor import FaceExtractor
from predictions.face_recognizer import FaceRecognizer
from predictions.trainers.svm_trainer import SVMTrainer
from analysis_config import configs, DatasetModifications
from settings import output

logging.basicConfig(filename="masked_face_recognizer.log")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")


class Analyzer:
    def __init__(self):
        self._dsc_builder = DatasetConfigBuilder()
        self._dataset_reader = DatasetReader()
        self._dataset_modifier = DatasetModifier()
        self._face_extractor = FaceExtractor()

    def read_dataset(self, path: str, n: int = 1) -> pd.DataFrame:
        return self._dataset_reader.read(self._dsc_builder.build(path), n)

    def reset(self) -> None:
        self._face_extractor.reset()

    def run(
        self,
        dataset_path: str,
        dataset_split_ratio: float,
        train_mods: Optional[DatasetModifications],
        test_mods: Optional[DatasetModifications],
        personal_images_quantity: int,
    ):
        logger.info("Analysis started.")

        logger.info("1. Dataset reading stage.")
        dataset = self.read_dataset(dataset_path, personal_images_quantity)
        train_set, test_set = self._dataset_reader.split_dataset(
            dataset, ratio=dataset_split_ratio
        )

        logger.info("2. Extracting embeddings stage.")
        if train_mods is None:
            logger.info("Train set has not been modified.")
            train_mods = DatasetModifications(0.0, True, False)

        embs = self._face_extractor.extract(
            self._dataset_modifier.modify(train_set, *train_mods)
        )
        # self._face_extractor.save()
        # embs = "../face_vectors.pickle"

        logger.info("3. Model training stage")
        trainer = SVMTrainer(embs)
        model = trainer.train()
        label_coder = trainer.label_encoder
        # t.store_model()

        logger.info("4. Face recognition stage.")
        fr = FaceRecognizer(model, label_coder)
        if test_mods is None:
            logger.info("Test set has not been modified.")
            test_mods = DatasetModifications(0.0, True, False)

        results = fr.recognize(self._dataset_modifier.modify(test_set, *test_mods))

        logger.info("Analysis ended.")
        return results


if __name__ == "__main__":
    logger.info("Program started.")
    start = time.time()
    if not os.path.exists(output):
        os.makedirs(output)
    analyzer = Analyzer()

    with open(output / "analysis.csv", "w") as csvfile:
        fieldnames = [
            "root_dir",
            "dataset_split_ratio",
            "train_modifications",
            "test_modifications",
            "perfect_acc",
            "top5_acc",
        ]
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for c in configs:
            stats = analyzer.run(*c.values())
            writer.writerow([*c.values(), *stats["accuracy"].values()])
            analyzer.reset()
    logger.info("Program finished.")
    logger.info("--- %s minutes ---" % ((time.time() - start) / 60))
