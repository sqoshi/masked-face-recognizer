import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetModifier:
    """Allow to determine way of masking dataset."""

    @staticmethod
    def modify(
        df: pd.DataFrame,
        mask_ratio: float = 0.0,
        inplace: bool = True,
        skip_unknown: bool = False,
    ) -> pd.DataFrame:
        """
        Modifies Dataset by adding images with imposed masks or imposing mask on images inplace.

        :param df: dataframe with columns: filename, identity
        :param mask_ratio: 0.5 means that 50% of images from dataframe will be masked
        :param skip_unknown: removes images with identity name = unknown or NULL
        :param inplace: decides if modified images will be added as news rows to dataframe
            or modifying existing rows
        :return: dataframe with columns filename, identity, impose_mask:bool
        """
        new_df = df.assign(impose_mask=lambda _: False)
        if mask_ratio != 0.0:
            logger.info(f"Masked {mask_ratio * 100}% of images inplace={inplace}.")
            for ident in set(df["identity"]):
                ident_images = new_df.loc[new_df["identity"] == ident]
                n = round(len(ident_images.index) * mask_ratio)
                rows = ident_images.head(n)
                if inplace:
                    new_df.loc[rows.index, "impose_mask"] = True
                else:
                    rows["impose_mask"] = rows["impose_mask"].apply(lambda _: True)
                    new_df = new_df.append(rows)

        if skip_unknown:
            logger.info("Skipping `unknown` identities.")
            new_df = new_df.drop(
                new_df[(new_df.identity == "unknown") | (new_df.identity is None)].index
            )

        return new_df.sort_values("identity").reset_index(drop=True)
