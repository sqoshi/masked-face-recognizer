import logging
import pickle
from collections import namedtuple
from typing import Dict, List, Any

import dlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from predictions.embedder import Embedder
from predictions.face_detector import FaceDetector
from predictions.image import Image

logger = logging.getLogger(__name__)


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
    return max([fix_rect(r) for r in rectangles], key=lambda x: x.area())


def warn_detections(face_detections: dlib.rectangles) -> None:
    """Logs warnings about face detection abuses."""
    if len(face_detections) > 1:
        logger.warning(
            "Detected %i faces on image. The biggest surface face will be processed."
            % len(face_detections)
        )
        logger.info("Selecting face rectangle with biggest area.")
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

    def __init__(self) -> None:
        FaceDetector.__init__(self)  # explicit calls without super
        Embedder.__init__(self)
        self._embeddings = {"vectors": [], "classes": []}

    def reset(self) -> None:
        self._embeddings = {"vectors": [], "classes": []}

    def _upload_embeddings(self, identity: str, vector: NDArray[Any]) -> None:
        """Saving identity and embeddings to dictionary."""
        self._embeddings["vectors"].append(vector)
        self._embeddings["classes"].append(identity)

    def save(self, fn: str = "../face_vectors.pickle") -> None:
        """Saving embeddings as dictionary to file."""
        logger.info("Saving embeddings to %s." % fn)
        with open(fn, 'wb') as fw:
            pickle.dump(self._embeddings, fw, protocol=pickle.HIGHEST_PROTOCOL)

    def extract(self, df: pd.DataFrame) -> Dict[str, List[NDArray[Any]]]:
        """Extracting embeddings from loaded images."""
        logger.info("Extracting embeddings.")
        for index, row in df.iterrows():
            logger.info(f"Extracting (%s/%s) ...", index, len(df))

            img = Image(row['filename'], row['identity'])
            face_rectangles = self._detector(img.obj, 1)
            warn_detections(face_rectangles)

            if face_rectangles:
                rect = biggest_surface(face_rectangles)  # todo: adjust rectangle

                face_crop = crop(img, rect)
                if row['impose_mask']:
                    face_crop = img.get_masked((face_crop, "m!_" + str(img.path)))

                embeddings_vec = self.vector(face_crop).flatten()
                self._upload_embeddings(img.identity, embeddings_vec)

        return self._embeddings
