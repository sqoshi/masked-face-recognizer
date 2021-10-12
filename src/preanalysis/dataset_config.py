import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from collections import namedtuple

logger = logging.getLogger(__name__)

DatasetConfig = namedtuple("DatasetConfig", "directory_fp identities_fp description_fp")


class DatasetConfigBuilder:
    """Class is responsible for Dataset config setup."""

    @classmethod
    def find_info_files(cls, path: str) -> Tuple[Optional[str], Optional[str]]:
        """Looking for csv file with identities and database description file."""
        description_fp, identities_fp = None, None

        for entity in os.listdir(path):
            entity = Path(os.path.join(path, entity))
            if entity.is_file():
                entity_low = entity.name.lower()
                if "ident" in entity_low:
                    if entity_low.endswith(".csv") or entity_low.endswith(".txt"):
                        logger.info("Identities file '%s' found." % entity)
                        identities_fp = str(entity)
                elif entity_low.endswith(".md") or entity_low.endswith(".txt"):
                    logger.info("Description file '%s' found." % entity)
                    description_fp = str(entity)

        return identities_fp, description_fp

    def build(self, path: str) -> DatasetConfig:
        """Builds dataset config according to root directory structure."""
        logger.info("Building dataset config for '%s'" % path)
        identities_fp, description_fp = self.find_info_files(path)
        subdirs = [e for e in os.listdir(path) if Path(os.path.join(path, e)).is_dir()]

        if len(subdirs) == 0:
            logger.critical("Dataset '%s' directory structure error!" % path)
            sys.exit()

        images_dir = os.path.join(path, subdirs.pop()) if len(subdirs) == 1 else path

        dc = DatasetConfig(
            directory_fp=images_dir,
            identities_fp=identities_fp,
            description_fp=description_fp
        )

        logger.info("Dataset configuration = '%s'" % str(dc))

        return dc
