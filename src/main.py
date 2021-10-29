import logging
import os
import time

import coloredlogs

from analysis_config import AnalysisConfig, DatasetModifications
from analyzer import Analyzer
from settings import output

logging.basicConfig(filename="masked_face_recognizer.log")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

datasets = [
    # "/home/piotr/Documents/bsc-thesis/datasets/test"
    f'/home/{os.environ["USER"]}/Documents/bsc-thesis/datasets/original',
    # "/home/piotr/Documents/bsc-thesis/datasets/celeba"
]


class AnalysisConfigFactory:
    def __init__(self, path):
        self.dataset_path = path

    def default(self):
        """Config without modifications."""
        return AnalysisConfig(
            name="no_modifications",
            dataset_path=self.dataset_path,
            split_ratio=0.8,
        )

    def masked_test_set(self):
        """Masked test set"""
        return AnalysisConfig(
            name="masked_test_set",
            dataset_path=self.dataset_path,
            split_ratio=0.8,
            test_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=True
            )
        )

    def masked_train_set_masked_test_set(self, p: float):
        return AnalysisConfig(
            name=f"{int(p * 100)}%_masked_train_set",
            dataset_path=self.dataset_path,
            split_ratio=0.8,
            train_set_modifications=DatasetModifications(
                mask_ratio=p, inplace=True
            ),
            test_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=True
            )
        )

    def masked_extended_train_set_masked_test_set(self, p: float):
        return AnalysisConfig(
            name=f"{int(p * 100)}%_masked_extended_train_set",
            dataset_path=self.dataset_path,
            split_ratio=0.8,
            train_set_modifications=DatasetModifications(
                mask_ratio=p, inplace=False
            ),
            test_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=True
            )
        )

    def research_path(self):
        return [
            self.default(),
            self.masked_test_set(),
            self.masked_train_set_masked_test_set(0.2),
            self.masked_extended_train_set_masked_test_set(0.2),
            self.masked_train_set_masked_test_set(0.5),
            self.masked_extended_train_set_masked_test_set(0.5),
            self.masked_train_set_masked_test_set(0.7),
            self.masked_extended_train_set_masked_test_set(0.7),
        ]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    logger.info("Program started.")
    start = time.time()
    mkdir(output)
    analyzer = Analyzer()

    acf = AnalysisConfigFactory(datasets[0])

    for c in acf.research_path():
        stats = analyzer.run(c)
        subdir = output / c.get_dataset_name() / str(int(start)) / c.name
        mkdir(subdir)
        c.to_json(subdir / "analysis_config.json")
        analyzer.save_model_details(subdir / "model_config.json")
        stats.to_csv(subdir / "results.csv")
        analyzer.reset()

    logger.info("Program finished.")
    logger.info("--- %s minutes ---" % ((time.time() - start) / 60))
