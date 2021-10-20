import logging
import pickle
from typing import Dict, List, Any

import dlib
import pandas as pd
from numpy.typing import NDArray

from predictions.embedder import Embedder
from predictions.face_detector import FaceDetector

logger = logging.getLogger(__name__)


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
        for i, (vec, img) in enumerate(self.vector_generator(df, self.vector)):
            logger.info(f"Extracting (%s/%s) ...", i, len(df.index))
            embeddings_vec = vec.flatten()
            self._upload_embeddings(img.identity, embeddings_vec)

        return self._embeddings

# def extract(self, df: pd.DataFrame) -> Dict[str, List[NDArray[Any]]]:
#     """Extracting embeddings from loaded images."""
#     logger.info("Extracting embeddings.")
#     for i, row in df.iterrows():
#         logger.info(f"Extracting (%s/%s) ...", i, len(df.index))
#
#         img = Image(row['filename'], row['identity'])
#         face_rectangles = self._detector(img.obj, 1)
#         warn_detections(face_rectangles)
#
#         if face_rectangles:
#             rect = biggest_surface(face_rectangles)  # todo: adjust rectangle
#
#             face_crop = crop(img, rect)
#             if row['impose_mask']:
#                 face_crop = img.get_masked((face_crop, "m!_" + str(img.path)))
#
#             embeddings_vec = self.vector(face_crop).flatten()
#             self._upload_embeddings(img.identity, embeddings_vec)
#
#     return self._embeddings
