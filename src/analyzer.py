import json
import logging
import os

import pandas as pd

from research_configurators.analysis_config import AnalysisConfig
from preanalysis.dataset_utils.dataset_configuration.dataset_config_builder import (
    DatasetConfigBuilder,
)
from preanalysis.dataset_utils.dataset_modification.dataset_modificator import DatasetModifier
from preanalysis.dataset_utils.dataset_reading.dataset_reader import DatasetReader
from predictions.extractor import FaceExtractor
from predictions.face_recognizer import FaceRecognizer
from predictions.trainers.svm_trainer import SVMTrainer

logger = logging.getLogger(__name__)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Analyzer:
    def __init__(self):
        self._dsc_builder = DatasetConfigBuilder()
        self._dataset_reader = DatasetReader()
        self._dataset_modifier = DatasetModifier()
        self._face_extractor = FaceExtractor()
        self._model_info = {}
        self._results = {}

    def save_model_details(self, fp: str):
        if not self._model_info:
            logger.warning("Model info is empty.")

        with open(fp, "w+") as fw:
            json.dump(self._model_info, fw)

    def read_dataset(self, analysis_config: AnalysisConfig) -> pd.DataFrame:
        return self._dataset_reader.read(
            self._dsc_builder.build(analysis_config.dataset_path), analysis_config
        )

    def reset(self) -> None:
        self._model_info = {}
        self._results = {}
        self._face_extractor.reset()

    def run(self, analysis_config: AnalysisConfig):
        logger.info("Analysis started.")

        logger.info("1. Dataset reading stage.")
        dataset = self.read_dataset(analysis_config)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        train_set, test_set = self._dataset_reader.split_dataset(
            dataset, ratio=analysis_config.split_ratio
        )

        logger.info("2. Extracting embeddings stage.")
        embs = self._face_extractor.extract(
            self._dataset_modifier.modify(train_set, analysis_config.modifications.train),
            analysis_config.landmarks_detection
        )

        logger.info("3. Model training stage")
        trainer = SVMTrainer(embs, analysis_config.svm_config)
        model = trainer.train()
        self._model_info = trainer.get_model_details()
        label_coder = trainer.label_encoder

        logger.info("4. Face recognition stage.")
        fr = FaceRecognizer(model, label_coder)

        results = fr.recognize(
            self._dataset_modifier.modify(test_set, analysis_config.modifications.test),
            analysis_config.landmarks_detection
        )

        logger.info("Analysis ended.")
        self._results = results
        return results

    def save(self, subdir, config, save_csv=False) -> None:
        mkdir(subdir)
        config.to_json(subdir / "analysis_config.json")
        self.save_model_details(subdir / "model_config.json")
        if save_csv:
            self._results.to_csv(subdir / "results.csv")  # must be saved as json
        with open(subdir / "results.json", "w+") as f:
            f.write(self._results.to_json(orient='table'))
        # self._results.to_json(subdir / "results.json", orient='records', lines=True)
