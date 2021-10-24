import csv
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
            stats = analyzer.run(c)
            writer.writerow([*c.values(), *stats["accuracy"].values()])
            analyzer.reset()
    logger.info("Program finished.")
    logger.info("--- %s minutes ---" % ((time.time() - start) / 60))
