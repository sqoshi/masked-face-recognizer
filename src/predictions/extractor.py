import logging
import pickle
from typing import Dict, List

import dlib
import numpy as np
import pandas as pd

from predictions.embedder import Embedder
from predictions.face_detector import FaceDetector
from predictions.image import Image

logger = logging.getLogger(__name__)


# def rect_to_bb(rect: dlib.rectangle):
#     """Transform dlib rectangle to left,right cords and width, height."""
#     x = rect.left()
#     y = rect.top()
#     w = rect.right() - x
#     h = rect.bottom() - y
#     return x, y, w, h


def fix_rect(rect: dlib.rectangle):
    """Replaces negative coordinates by 0."""
    return dlib.rectangle(
        top=nn(rect.top()),
        bottom=nn(rect.bottom()),
        left=nn(rect.left()),
        right=nn(rect.right())
    )


def biggest_surface(rectangles: dlib.rectangles) -> dlib.rectangle:
    """Selects rectangle with biggest area."""
    logger.info("Selecting face rectangle with biggest area.")
    return max([fix_rect(r) for r in rectangles], key=lambda x: x.area())


def warn_detections(face_detections: dlib.rectangles) -> None:
    """Logs warnings about face detection abuses."""
    if len(face_detections) > 1:
        logger.warning(
            f"Detected {len(face_detections)} faces on image. The biggest surface face will be processed.")
    elif len(face_detections) == 0:
        logger.warning("Could not detect face on image.")


def nn(value: int) -> int:
    """Casts value to closest non negative value"""
    return 0 if value < 0 else value


def crop(image: Image, rect: dlib.rectangle) -> np.ndarray:
    """Cuts image to rectangle coordinates."""
    return image.obj[rect.top():rect.bottom(), rect.left():rect.right()]


class FaceExtractor(FaceDetector, Embedder):
    """Detecting face on image and transforms it to vector."""

    def __init__(self, dataset_df: pd.DataFrame) -> None:
        FaceDetector.__init__(self)  # explicit calls without super
        Embedder.__init__(self)
        self.dataset_df = dataset_df.reset_index(drop=True)
        self._embeddings = {"vectors": [], "classes": []}

    def save(self, fn: str = "../face_vectors.pickle") -> None:
        """Saving embeddings as dictionary to file."""
        logger.info("Saving embeddings to %s." % fn)
        with open(fn, 'wb') as fw:
            pickle.dump(self._embeddings, fw, protocol=pickle.HIGHEST_PROTOCOL)

    def extract(self) -> Dict[str, List[np.ndarray]]:
        """Extracting embeddings from loaded images."""
        logger.info("Extracting embeddings.")
        for index, row in self.dataset_df.iterrows():
            logger.info(f"Extracting (%s/%s) ...", index, len(self.dataset_df))
            img = Image(row['filename'], row['identity'])
            face_rectangles = self._detector(img.obj, 1)  # get masked
            warn_detections(face_rectangles)
            if face_rectangles:
                rect = biggest_surface(face_rectangles)  # todo: adjust rectangle
                face_crop = crop(img, rect)
                embeddings_vec = self.vector(face_crop).flatten()
                self._embeddings["vectors"].append(embeddings_vec)
                self._embeddings["classes"].append(img.identity)
        return self._embeddings
