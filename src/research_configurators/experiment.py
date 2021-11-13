from typing import List

from research_configurators.analysis_config import AnalysisConfig


class Experiment:
    def __init__(self, name: str, configs: List[AnalysisConfig],
                 description: str = "") -> None:
        self.name = name
        self.description = description
        self.experiment_path = configs

    def __iter__(self):
        for x in self.experiment_path:
            yield x
