import logging
import os
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from imutils.paths import list_images
from termcolor import cprint

from preanalysis.defs import DatasetConfig

logger = logging.getLogger(__name__)


def draw_bar_diagram(df: pd.DataFrame) -> None:
    """Images per identity visualization."""
    df["identity"].value_counts().plot(kind='bar')
    plt.show()


class DatasetReader:

    def __init__(self, dataset_config: DatasetConfig, min_images_per_person=25) -> None:
        self.images_per_person = min_images_per_person
        self.dataset_path = dataset_config.directory_fp
        self.column_names = ["filename", "identity"]
        logger.info("Dataset path '%s'" % self.dataset_path)

        # count images per person
        if dataset_config.identities_fp:
            logger.info("Identities file found.")
            self.df = pd.read_csv(dataset_config.identities_fp, sep=" ", index_col=False, names=self.column_names)
            self.statistics = self.df["identity"].value_counts()[
                (self.df["identity"].value_counts() >= min_images_per_person)
            ]
            self.identities = self.statistics.keys().tolist()
            self.filtered = self.df.loc[self.df["identity"].isin(self.identities)].copy()
            self.filtered["filename"] = self.filtered["filename"].apply(
                lambda x: os.path.join(self.dataset_path, str(x)))
            if self.statistics.max() < min_images_per_person:
                cprint(f"There is not enough images per person for your demand `{min_images_per_person}`."
                       f" Max is {self.statistics.max()}", "yellow")
        else:
            logger.info("Identities collected from directories names.")
            images = list(list_images(dataset_config.directory_fp))
            self.identities = [path.split(os.sep)[-2] for path in images]
            self.filtered = pd.DataFrame(list(zip(images, self.identities)), columns=self.column_names)

    def select_images(self, n: int) -> pd.DataFrame:
        """ Select n images per person from Dataset.

        Datasets may contain different quantity of images per identity.
        identity X = 5 images, identity Y = 12 images.
        If user want to preserve equality of images per person, this method allow to select
        n images per identity/person from Dataset. """

        result = None
        for ident in set(self.identities):
            personal_images = self.filtered.loc[self.filtered["identity"] == ident].head(n)
            if result is None:
                result = personal_images
            else:
                result = pd.concat([result, personal_images], ignore_index=True)

        if len(set(result["identity"].value_counts().values.tolist())) != 1:
            logger.warning(f"Something went wrong! You chose `equality` option,"
                           f" but personal images quantities are not equal.", "yellow")

        return result

    @staticmethod
    def split_dataset(dataset_df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Splitting dataset into train and tests subsets with ratio = %s.", ratio)
        images_per_person = min(dataset_df["identity"].value_counts().values.tolist())
        train_limit = round(images_per_person * ratio)
        train_set = dataset_df.groupby("identity").head(train_limit)
        test_set = dataset_df[~dataset_df.index.isin(train_set.index)]
        return train_set, test_set

    def read(self, images_per_person: Optional[int] = None, equalize: bool = True) -> pd.DataFrame:

        if equalize:
            logger.info("Reading %s per person." % images_per_person)
            self.filtered = self.select_images(
                self.images_per_person if images_per_person is None else images_per_person
            )

        logger.info(f"Dataset consist %s identities." % len(self.filtered.groupby('identity')))
        # draw_bar_diagram(self.filtered)
        return self.filtered
