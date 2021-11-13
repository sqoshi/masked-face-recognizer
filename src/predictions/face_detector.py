import random
from typing import Any, Callable, Tuple

import dlib
import numpy as np
import pandas as pd
from _dlib_pybind11 import get_frontal_face_detector
from numpy.typing import NDArray

from research_configurators.mask_strategy import MaskingStrategy
from predictions.image import Image


def nn(value: int) -> int:
    """Casts value to closest non negative value"""
    return 0 if value < 0 else value


def fix_rect(rect: dlib.rectangle):
    """Replaces negative coordinates by 0."""
    return dlib.rectangle(
        top=nn(rect.top()),
        bottom=nn(rect.bottom()),
        left=nn(rect.left()),
        right=nn(rect.right()),
    )


def biggest_surface(rectangles: dlib.rectangles) -> dlib.rectangle:
    """Selects rectangle with biggest area."""
    return max([fix_rect(r) for r in rectangles], key=lambda x: x.area())


def crop(image: Image, rect: dlib.rectangle) -> np.ndarray:
    """Cuts image to rectangle coordinates."""
    return image.obj[rect.top(): rect.bottom(), rect.left(): rect.right()]


class FaceDetector:
    """Contains detector from `dlib`."""

    def __init__(self) -> None:
        self._detector = get_frontal_face_detector()
        self._real_masks = [MaskingStrategy.blue, MaskingStrategy.grey]
        self._masks = self._real_masks + [MaskingStrategy.black_box]
        self._last_mask = random.choice(self._real_masks)

    def vector_generator(
            self, df: pd.DataFrame, vector_func: Callable
    ) -> Tuple[NDArray[Any], Image]:
        """Creates generator which yields vectors created by openface from images and this image."""

        for index, row in df.iterrows():
            img = Image(row["filename"], row["identity"])
            face_rectangles = self._detector(img.obj, 1)

            if face_rectangles:
                rect = biggest_surface(face_rectangles)
                face_crop = crop(img, rect)

                if row["impose_mask"]:
                    if row["impose_mask"] == MaskingStrategy.alternately:
                        img.switch_mask_imposer_mask(self._last_mask.value)
                        self._last_mask = [x for x in self._real_masks if x != self._last_mask][0]
                    elif row["impose_mask"] in self._masks:
                        img.switch_mask_imposer_mask(row["impose_mask"])
                    face_crop = img.get_masked((face_crop, "m!_" + str(img.path)))

                yield vector_func(face_crop), img
