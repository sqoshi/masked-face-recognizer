import logging
import os
from enum import Enum
from typing import Callable, List, Optional

import pandas as pd
from imutils.paths import list_images
from numpy import isnan

from preanalysis.dataset_utils.dataset_configuration.dataset_config_builder import DatasetConfig

logger = logging.getLogger(__name__)


class Strategy(Enum):
    """Represents structure of images kept under dataset directory path."""

    grouped = "grouped"
    mixed = "mixed"  # requires identities file


def _limit_identities(identities_list: List[str], identities_limit: Optional[int]) -> List[str]:
    """Limits identities list to identities_limit if needed."""
    if identities_limit is not None:
        if identities_limit < 2:
            raise ValueError(f"Identities limit must >=2. (identities_limit={identities_limit})")
        return identities_list[:identities_limit]
    return identities_list


def _limit_dataframe(
        dataset_df: pd.DataFrame, piq: int, identities_limit: Optional[int] = None,
        is_piq_max: bool = True
) -> pd.DataFrame:
    """Limits identities images to exactly/ at least `piq` images per identity and
        limits identities quantity to `identities_limit`

    :param dataset_df: dataframe with all images.
    :param piq: personal images quantity
    :param is_piq_max: if false then `personal_images_quantity` is treated as
                        'at least' images per person, else `exactly`

    :return: dataframe with exactly or at least `personal_images_quantity` images per person.
    """
    # select only images with at least n personal images.
    piq_table = dataset_df["identity"].value_counts()
    piq_table = piq_table[piq_table >= piq]

    if isnan(piq_table.max()) or piq_table.max() < piq:
        raise ValueError(
            f"There is not enough ({piq}) images for any identity. Max is {piq_table.max()}."
        )

    identities = _limit_identities(piq_table.keys().tolist(), identities_limit)

    new_df = dataset_df.loc[dataset_df["identity"].isin(identities)].copy()
    if is_piq_max:
        new_df = new_df.groupby("identity").head(piq)
    return new_df.reset_index(drop=True)


def auto_strategy(dc: DatasetConfig) -> Strategy:
    """Selects strategy depending on identities file existence."""
    return Strategy.grouped if dc.identities_fp is None else Strategy.mixed


class ReaderFactory:
    """Accordingly to input dataset config reading dataset with appropriate strategy with or
    without limitations.
    """

    def __init__(self, columns: List[str] = ("filename", "identity")) -> None:
        self.columns = columns

    def read(self, dc: DatasetConfig, n: int, strategy: Strategy = None) -> pd.DataFrame:
        """Read dataset with n images per identity with special strategy."""
        strategy = auto_strategy(dc) if strategy is None else strategy
        reader = self._get_reader(strategy)
        return reader(dc, n)

    def _get_reader(self, strategy: Strategy = None) -> Callable:
        """Selects reading methodology by strategy."""
        if strategy == Strategy.grouped:
            return self._read_grouped
        if strategy == Strategy.mixed:
            return self._read_mixed
        raise ValueError(strategy)

    def _read_grouped(self, dc: DatasetConfig, n: int = 1) -> pd.DataFrame:
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
        dataset_df = _limit_dataframe(dataset_df, n)
        return dataset_df

    def _read_mixed(self, dc: DatasetConfig, n: int = 1) -> pd.DataFrame:
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
        dataset_df = _limit_dataframe(attrs_df, n)
        dataset_df["filename"] = dataset_df["filename"].apply(
            lambda x: os.path.join(dc.directory_fp, str(x))
        )
        return dataset_df
