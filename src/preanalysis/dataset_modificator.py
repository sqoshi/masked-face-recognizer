import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetModifier:
    """Allow to determine way of masking dataset."""

    @staticmethod
    def modify(
            df: pd.DataFrame, mask_ratio: float = 0.0, inplace: bool = True
    ) -> pd.DataFrame:
        """
        Modifies Dataset.
        :param df: dataframe with columns: filename, identity
        :param mask_ratio: 0.5 means that 50% of images from dataframe will be masked
        :param inplace: decides if modified images will be added as news rows to dataframe
            or modifying existing rows
        :return: dataframe with columns filename, identity, impose_mask:bool
        """
        new_df = df.assign(impose_mask=lambda _: False)
        if mask_ratio != 0.0:
            for ident in set(df["identity"]):
                ident_images = new_df.loc[new_df["identity"] == ident]
                n = round(len(ident_images.index) * mask_ratio)
                logger.info(f"Masked {mask_ratio * 100}% ({n}) of images.")
                rows = ident_images.head(n)
                if inplace:
                    new_df.loc[rows.index, "impose_mask"] = True
                else:
                    rows["impose_mask"] = rows["impose_mask"].apply(lambda _: True)
                    new_df = new_df.append(rows)

        new_df = new_df.sort_values("identity").reset_index(drop=True)
        return new_df
