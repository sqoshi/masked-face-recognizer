import logging

import pandas as pd

from config.run_configuration import DatasetModifications

logger = logging.getLogger(__name__)


class DatasetModifier:
    """Allow to determine way of masking dataset."""

    @staticmethod
    def modify(
        df: pd.DataFrame,
        modifications: DatasetModifications,
    ) -> pd.DataFrame:
        """
        Modifies Dataset by adding images with imposed masks or imposing mask on images inplace.

        :param df: dataframe with columns: filename, identity
        :param modifications:
            - modifications.mask_ratio: 0.5 means that 50% of images from dataframe will be masked
            - modifications.skip_unknown: removes images with identity name = unknown or NULL
            - modifications.inplace: decides if modified images will be added as news rows to
                dataframe
            or modifying existing rows
        :return: dataframe with columns filename, identity, impose_mask:bool
        """
        new_df = df.assign(impose_mask=lambda _: False)
        if modifications.mask_ratio != 0.0:
            logger.info(
                f"Masked {modifications.mask_ratio * 100}% of images,"
                f" inplace={modifications.inplace}."
            )
            for ident in set(df["identity"]):
                ident_images = new_df.loc[new_df["identity"] == ident]
                n = round(len(ident_images.index) * modifications.mask_ratio)
                rows = ident_images.head(n)
                if modifications.inplace:
                    new_df.loc[rows.index, "impose_mask"] = modifications.mask
                else:
                    rows["impose_mask"] = rows["impose_mask"].apply(
                        lambda _: modifications.mask
                    )
                    new_df = new_df.append(rows)

        # if modifications.skip_unknown:
        #     logger.info("Skipping `unknown` identities.")
        #     new_df = new_df.drop(
        #         new_df[(new_df.identity == "unknown") | (new_df.identity is None)].index
        #     )

        return new_df.sort_values("identity").reset_index(drop=True)
