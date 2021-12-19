import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from preanalysis.dataset_utils.dataset_configuration.dataset_config_builder import (
    DatasetConfigBuilder,
)
from preanalysis.dataset_utils.dataset_modification.dataset_modificator import (
    DatasetModifier,
)
from preanalysis.dataset_utils.dataset_reading.dataset_reader import DatasetReader
from predictions.extractor import FaceExtractor
from predictions.face_recognizer import FaceRecognizer
from predictions.trainers.svm_trainer import SVMTrainer
from config.run_configuration import Configuration

logger = logging.getLogger(__name__)


def mkdir(path: Union[Path, str]) -> None:
    """Creates directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


class Runner:
    """Class is responsible for performing whole procedure
    from reading dataset to model evaluation.
    """

    def __init__(self) -> None:
        self._dsc_builder = DatasetConfigBuilder()
        self._dataset_reader = DatasetReader()
        self._dataset_modifier = DatasetModifier()
        self._face_extractor = FaceExtractor()
        self._model_info: Dict[Any, Any] = {}
        self._results: Dict[Any, Any] = {}
        self._model = None

    def read_dataset(self, analysis_config: Configuration) -> pd.DataFrame:
        """Read dataset with appropriate configuration."""
        return self._dataset_reader.read(
            self._dsc_builder.build(analysis_config.dataset_path), analysis_config
        )

    def reset(self) -> None:
        """Resets memory after run procedure."""
        self._model_info = {}
        self._results = {}
        self._face_extractor.reset()

    def run(self, analysis_config: Configuration) -> pd.DataFrame:
        """Starts learning and evaluating procedures.

        1. read dataset with appropriate limitations
        3. split dataset into train and test set
        2. modify images as in input config
        3. extract embeddings from train set
        4. train model with embeddings ( train set )
        5. evaluate model on images from test set
        6. save config and model
        """
        logger.info("Analysis started.")

        logger.info("1. Dataset reading stage.")
        dataset = self.read_dataset(analysis_config)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        train_set, test_set = self._dataset_reader.split_dataset(
            dataset, ratio=analysis_config.split_ratio
        )

        logger.info("2. Extracting embeddings stage.")
        embs = self._face_extractor.extract(
            self._dataset_modifier.modify(
                train_set, analysis_config.modifications.train
            ),
            analysis_config.landmarks_detection,
        )

        logger.info("3. Model training stage")
        trainer = SVMTrainer(embs, analysis_config.svm_config)
        self._model = trainer.train()
        self._model_info = trainer.get_model_details()

        logger.info("4. Face recognition stage.")
        fr = FaceRecognizer(self._model, trainer.label_encoder)

        results = fr.recognize(
            self._dataset_modifier.modify(test_set, analysis_config.modifications.test),
            analysis_config.landmarks_detection,
        )

        logger.info("Analysis ended.")
        self._results = results
        return results

    def save_model_details(self, fp: Union[Path, str]) -> None:
        """Save model structure to file."""
        if not self._model_info:
            logger.warning("Model info is empty.")

        with open(fp, "w+") as fw:  # pylint: disable=unspecified-encoding
            json.dump(self._model_info, fw)

    def save(self, subdir: Path, config: Configuration, save_csv: bool = False) -> None:
        """Saves dataset modifications, learning method, trained model and
        counted classifications.
        """
        mkdir(subdir)
        # save modifications
        config.to_json(subdir / "analysis_config.json")

        # save trained model
        with open(subdir / "model.pkl", "wb") as f:
            pickle.dump(self._model, f)

        # save model structure info
        self.save_model_details(subdir / "model_config.json")

        # save counted classifications
        if save_csv:
            self._results.to_csv(subdir / "results.csv")  # type: ignore
        with open(subdir / "results.json", "w+") as f:  # type: ignore
            f.write(self._results.to_json(orient="table"))  # type: ignore
        # self._results.to_json(subdir / "results.json", orient='records', lines=True)
