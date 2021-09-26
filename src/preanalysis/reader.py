import logging
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

from preanalysis.defs import DatasetConfig
from preanalysis.reader_factory import ReaderFactory

logger = logging.getLogger(__name__)


def draw_bar_diagram(df: pd.DataFrame) -> None:
    """Images per identity visualization."""
    df["identity"].value_counts().plot(kind='bar')
    plt.show()


class DatasetReader:
    """Reads dataset with limitations and allows to split dataset."""

    def __init__(self, dataset_config: DatasetConfig) -> None:
        logger.info("Target %s." % str(dataset_config))
        self.columns = ("filename", "identity")
        self._ds_config = dataset_config
        self._reader_factory = ReaderFactory()

    @staticmethod
    def split_dataset(dataset_df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits dataset to train and test set."""
        print(dataset_df)
        logger.info("Splitting dataset into train and tests subsets with ratio = %s.", ratio)
        images_per_person = min(dataset_df["identity"].value_counts().values.tolist())
        train_limit = round(images_per_person * ratio)
        logger.info("images per identity - Train: %s, Test: %s", train_limit, images_per_person - train_limit)
        train_set = dataset_df.groupby("identity").head(train_limit)
        test_set = dataset_df[~dataset_df.index.isin(train_set.index)]
        return train_set, test_set

    def read(self, n: int = 1) -> pd.DataFrame:
        """Reads n images per identity using reader factory."""
        return self._reader_factory.read(self._ds_config, n)
