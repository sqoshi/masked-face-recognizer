import logging
import os
from enum import Enum
from typing import Callable, List

import pandas as pd
from imutils.paths import list_images

from analysis_config import AnalysisConfig
from preanalysis.dataset_utils.dataset_configuration.dataset_config_builder import DatasetConfig
from preanalysis.dataset_utils.dataset_reading.reading_limitations import limit_dataframe

logger = logging.getLogger(__name__)


class Strategy(Enum):
    """Represents structure of images kept under dataset directory path."""

    grouped = "grouped"
    mixed = "mixed"  # requires identities file


def auto_strategy(dc: DatasetConfig) -> Strategy:
    """Selects strategy depending on identities file existence."""
    return Strategy.grouped if dc.identities_fp is None else Strategy.mixed


class ReaderFactory:
    """Accordingly to input dataset config reading dataset with appropriate strategy with or
    without limitations.
    """

    def __init__(self, columns: List[str] = ("filename", "identity")) -> None:
        self.columns = columns

    def read(self, dc: DatasetConfig, analysis_config:AnalysisConfig, strategy: Strategy = None) -> pd.DataFrame:
        """Read dataset with n images per identity with special strategy."""
        strategy = auto_strategy(dc) if strategy is None else strategy
        reader = self._get_reader(strategy)
        return reader(dc, analysis_config)

    def _get_reader(self, strategy: Strategy = None) -> Callable:
        """Selects reading methodology by strategy."""
        if strategy == Strategy.grouped:
            return self._read_grouped
        if strategy == Strategy.mixed:
            return self._read_mixed
        raise ValueError(strategy)

    def _read_grouped(self, dc: DatasetConfig, analysis_config:AnalysisConfig) -> pd.DataFrame:
        """Reading n images per identity in identity-grouped structure.

        Example structure:
            dataset
            ├── andrzej_duda
            │         ├── 0.png
            │         └── 1.png
            │
            ├── andrzej_stefaniak
            │         ├── 0.png
            │         └── 1.png
            │
            ...

        """
        logger.info("Reading dataset grouped by identity images in subdirectory.")
        images = list(list_images(dc.directory_fp))
        identities = [path.split(os.sep)[-2] for path in images]
        dataset_df = pd.DataFrame(list(zip(images, identities)), columns=self.columns)
        dataset_df = limit_dataframe(dataset_df, analysis_config)
        return dataset_df

    def _read_mixed(self, dc: DatasetConfig, analysis_config:AnalysisConfig) -> pd.DataFrame:
        """Reading n images per identity in mixed images directory.

        Example structure:
            dataset
            ├── attributes.csv (must contain columns identity, filename)
            ├── 0.png
            ├── 1.png
            ├── 2.png
            ...
        """
        logger.info("Reading dataset mixed with identities in .csv file.")
        attrs_df = pd.read_csv(dc.identities_fp, sep=" ", index_col=False, names=self.columns)
        dataset_df = limit_dataframe(attrs_df, analysis_config)
        dataset_df["filename"] = dataset_df["filename"].apply(
            lambda x: os.path.join(dc.directory_fp, str(x))
        )
        return dataset_df
