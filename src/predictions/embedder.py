from typing import Any, Tuple

import cv2
from numpy.typing import NDArray


class Embedder:
    """Class is responsible for transforming face images to vectors using `openface` model."""

    def __init__(
        self,
        embedder_fp: str = "../models/nn4.small2.v1.t7",
        input_shape: Tuple[int, int] = (96, 96),
    ):
        self._embedder = cv2.dnn.readNetFromTorch(embedder_fp)
        self._embedder_input_shape = input_shape

    def vector(self, face_crop: NDArray[Any]) -> NDArray[Any]:
        """Creates 128-D vector using `openface` model.

        Takes matrix with cropped face and creates blob.
        Passes blob to embedder which produces 128-d vector.
        """
        face_blob = cv2.dnn.blobFromImage(
            face_crop,
            1.0 / 255,
            self._embedder_input_shape,
            (0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self._embedder.setInput(face_blob)
        vec = self._embedder.forward()
        return vec
