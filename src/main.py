import logging
import os
import time
from typing import List, Union

import coloredlogs

from runner import Runner
from config.run_configuration import Configuration
from config.configurator import Configurator
from config.grouped_config import GroupConfiguration
from settings import output

logging.basicConfig(filename="masked_face_recognizer.log")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")


def mkdir(path: str) -> None:
    """Create directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def investigate(
    analysis_list: List[Union[GroupConfiguration, Configuration]],
    research_group: str,
    subgroup_dir: str = "",
) -> None:
    """Running learning procedure for each element in list."""
    for config in analysis_list:
        if isinstance(config, Configuration):
            subdir = (
                output
                / config.get_dataset_name()
                / research_group
                / subgroup_dir
                / config.name
            )
            analyzer.run(config)
            analyzer.save(subdir, config)
            analyzer.reset()
        elif isinstance(config, GroupConfiguration):
            investigate(config, research_group, subgroup_dir=config.name)


if __name__ == "__main__":
    logger.info("Program started.")
    start = time.time()

    mkdir(output)
    datasets = [
        # "/home/piotr/Documents/bsc-thesis/datasets/test"
        f'/home/{os.environ["USER"]}/Documents/bsc-thesis/datasets/original',
        f'/home/{os.environ["USER"]}/Documents/bsc-thesis/datasets/celeba',
    ]

    analyzer = Runner()

    configurator = Configurator(datasets[0])  # pass path to your dataset here
    configurator.push(configurator.default())  # push config to a configurator, examples in class

    investigate(configurator.get_config_queue(), str(int(start)))

    logger.info("Program finished.")
    logger.info("--- %s minutes ---" % ((time.time() - start) / 60))
