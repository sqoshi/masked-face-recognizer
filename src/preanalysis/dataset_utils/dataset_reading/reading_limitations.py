import logging
from typing import List, Optional

import pandas as pd
from numpy import isnan

logger = logging.getLogger(__name__)


def _limit_identities(identities_list: List[str], identities_limit: Optional[int]) -> List[str]:
    """Limits identities list to identities_limit if needed."""
    if identities_limit is not None:
        if identities_limit < 2:
            raise ValueError(f"Identities limit must >=2. (identities_limit={identities_limit})")
        return identities_list[:identities_limit]

    return identities_list


def filter_dataframe_groups(df: pd.DataFrame, minimal_quantity: int) -> pd.DataFrame:
    """Filters dataframe by identities with at least `minimal_quantity` images."""
    personal_images_quantities = df["identity"].value_counts()
    return personal_images_quantities[personal_images_quantities >= minimal_quantity]


def limit_dataframe(
    dataset_df: pd.DataFrame,
    piq: int,
    identities_limit: Optional[int] = None,
    is_piq_max: bool = True,
) -> pd.DataFrame:
    """Limits identities images to exactly/ at least `piq` images per identity and
        limits identities quantity to `identities_limit`

    :param dataset_df: dataframe with all images.
    :param piq: personal images quantity
    :param identities_limit: quantity of identities to read
    :param is_piq_max: if false then `personal_images_quantity` is treated as
                        'at least' images per person, else `exactly`

    :return: dataframe with exactly or at least `personal_images_quantity` images per person.
    """
    # select only images with at least n personal images.
    piq_max = dataset_df["identity"].value_counts().max()
    piqs_table = filter_dataframe_groups(dataset_df, piq)

    if isnan(piqs_table.max()) or piqs_table.max() < piq:
        logger.critical(
            f"There is not enough ({piq}) images for any identity." f" Max is {piq_max}."
        )
        piqs_table = filter_dataframe_groups(dataset_df, piq_max)

    logger.info(f"Reading {piq} images per person.")

    identities = _limit_identities(piqs_table.keys().tolist(), identities_limit)
    logger.info(f"Reading images for {len(identities)} identities.")

    new_df = dataset_df.loc[dataset_df["identity"].isin(identities)].copy()
    if is_piq_max:
        new_df = new_df.groupby("identity").head(piq)
    return new_df.reset_index(drop=True)
