import logging
import os
import time

import coloredlogs

from research_configurators.analysis_config import AnalysisConfig
from research_configurators.analysis_config_factory import AnalysisConfigFactory
from analyzer import Analyzer
from research_configurators.experiment import Experiment
from settings import output

logging.basicConfig(filename="masked_face_recognizer.log")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

datasets = [
    # "/home/piotr/Documents/bsc-thesis/datasets/test"
    f'/home/{os.environ["USER"]}/Documents/bsc-thesis/datasets/original',
    f'/home/{os.environ["USER"]}/Documents/bsc-thesis/datasets/celeba'
]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def investigate(analysis_list, research_group, subgroup_dir=""):
    for config in analysis_list:
        if isinstance(config, AnalysisConfig):
            subdir = output / config.get_dataset_name() / research_group / subgroup_dir / config.name
            analyzer.run(config)
            analyzer.save(subdir, config)
            analyzer.reset()
        elif isinstance(config, Experiment):
            investigate(config, research_group, subgroup_dir=config.name)


if __name__ == "__main__":
    logger.info("Program started.")
    start = time.time()
    mkdir(output)
    analyzer = Analyzer()

    acf = AnalysisConfigFactory(datasets[-1])

    investigate(acf.research_path(), str(int(start)))

    logger.info("Program finished.")
    logger.info("--- %s minutes ---" % ((time.time() - start) / 60))

# researches list:
# 1. default - no modifications
# 2. masked test set:
#   1. influence of masked images ratio in train set with same mask [0.2,0.5,0.7,1.0]
#   2. influence of extending train set by masked images / modifying inplace [inplace vs extended]
#   3. influence of adding unknown identity [ skip unknown personality ]
#   4. influence of mixing different masks [alternately(grey, blue)]
#   5. influence of using black boxes instead of masks
#   6. influence of extracting embeddings only from characteristic points. [must be in place] <- #TODO[blob]
#   7. influence of extracting embeddings only from  whole image.
#
