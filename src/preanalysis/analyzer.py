import os
from typing import Union, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from imutils.paths import list_images
from termcolor import cprint

from src.preanalysis.dataclasses import DatasetConfig


def draw_bar_diagram(df: pd.DataFrame) -> None:
    """Images per identity visualization."""
    df["identity"].value_counts().plot(kind='bar')
    plt.show()


class DatasetReader:

    def __init__(self, dataset_config: DatasetConfig, min_images_per_person=25) -> None:
        self.images_per_person = min_images_per_person
        self.dataset_path = dataset_config.directory_fp
        self.column_names = ["filename", "identity"]

        # count images per person
        if dataset_config.identities_fp:
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
            images = list(list_images(dataset_config.directory_fp))
            self.identities = [path.split(os.sep)[-2] for path in images]
            print(len(images), len(self.identities))
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
            cprint(f"Something went wrong! You chose `equality` option,"
                   f" but personal images quantities are not equal.", "yellow")

        return result

    def read(self, images_per_person: Optional[int] = None, equalize: bool = True) -> pd.DataFrame:
        if equalize:
            self.filtered = self.select_images(
                self.images_per_person if images_per_person is None else images_per_person
            )
        cprint(f"Dataset consist {len(self.filtered.groupby('identity'))} identities.", "green")
        draw_bar_diagram(self.filtered)
        print(self.filtered.sort_values("identity"))
        return self.filtered
