import logging

import pandas as pd

from analysis_config import AnalysisConfig
from preanalysis.dataset_utils.dataset_configuration.dataset_config_builder import (
    DatasetConfigBuilder,
)
from preanalysis.dataset_utils.dataset_modification.dataset_modificator import DatasetModifier
from preanalysis.dataset_utils.dataset_reading.dataset_reader import DatasetReader
from predictions.extractor import FaceExtractor
from predictions.face_recognizer import FaceRecognizer
from predictions.trainers.svm_trainer import SVMTrainer

logger = logging.getLogger(__name__)


class Analyzer:
    def __init__(self):
        self._dsc_builder = DatasetConfigBuilder()
        self._dataset_reader = DatasetReader()
        self._dataset_modifier = DatasetModifier()
        self._face_extractor = FaceExtractor()

    def read_dataset(self, analysis_config: AnalysisConfig) -> pd.DataFrame:
        return self._dataset_reader.read(self._dsc_builder.build(analysis_config.dataset_path), analysis_config)

    def reset(self) -> None:
        self._face_extractor.reset()

    def run(self, analysis_config: AnalysisConfig):
        logger.info("Analysis started.")

        logger.info("1. Dataset reading stage.")
        dataset = self.read_dataset(analysis_config)
        train_set, test_set = self._dataset_reader.split_dataset(
            dataset, ratio=analysis_config.split_ratio
        )

        logger.info("2. Extracting embeddings stage.")
        embs = self._face_extractor.extract(
            self._dataset_modifier.modify(train_set, analysis_config.modifications.train)
        )

        logger.info("3. Model training stage")
        trainer = SVMTrainer(embs)
        model = trainer.train()
        label_coder = trainer.label_encoder

        logger.info("4. Face recognition stage.")
        fr = FaceRecognizer(model, label_coder)

        results = fr.recognize(
            self._dataset_modifier.modify(test_set, analysis_config.modifications.test)
        )

        logger.info("Analysis ended.")
        return results
