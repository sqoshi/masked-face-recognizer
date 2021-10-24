import logging
import os
import time

import coloredlogs

from analysis_config import AnalysisConfig
from analyzer import Analyzer
from settings import output

logging.basicConfig(filename="masked_face_recognizer.log")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

datasets = [
    # v"/home/piotr/Documents/bsc-thesis/datasets/test"
    "/home/piotr/Documents/bsc-thesis/datasets/original"
    # "/home/piotr/Documents/bsc-thesis/datasets/celeba"
]

configs = [
    AnalysisConfig(dataset_path=datasets.pop(),
                   split_ratio=0.8,
                   personal_images_quantity=6,
                   is_piq_max=True,
                   identities_limit=3),
    # (standard_path, 0.8, None, DatasetModifications(1.0, True, False)),
    # (standard_path, 0.8, DatasetModifications(0.2, True, False), DatasetModifications(1.0, True, False)),
    # (standard_path, 0.8, DatasetModifications(0.2, False), DatasetModifications(1.0, True)),
    # (standard_path, 0.8, DatasetModifications(0.5, True), DatasetModifications(1.0, True)),
    # (standard_path, 0.8, DatasetModifications(0.5, False), DatasetModifications(1.0, True)),
    # (standard_path, 0.8, DatasetModifications(1.0, True), DatasetModifications(1.0, True)),
    # (standard_path, 0.8, DatasetModifications(1.0, False), DatasetModifications(1.0, True)),
]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    logger.info("Program started.")
    start = time.time()
    mkdir(output)
    analyzer = Analyzer()

    for c in configs:
        stats = analyzer.run(c)
        subdir = output / c.get_dataset_name() / c.name
        mkdir(subdir)
        c.to_json(subdir / "analysis_config.json")
        analyzer.save_model_details(subdir / "model_config.json")
        stats.to_csv(subdir / "results.csv")
        analyzer.reset()

    logger.info("Program finished.")
    logger.info("--- %s minutes ---" % ((time.time() - start) / 60))
