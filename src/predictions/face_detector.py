import copy
import random
import sys
from typing import Any, Callable, Tuple

import cv2
import dlib
import numpy as np
import pandas as pd
from _dlib_pybind11 import get_frontal_face_detector
from numpy.typing import NDArray

from predictions.landmarks_predictor.landmark_detector.landmark_detector import LandmarksPredictor
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


def filter_eyes_landmarks(landmarks):
    """Collect eyes landmarks"""
    eyes_indexes = [_ for _ in range(37, 48)]
    out_arr = []
    for (k, v) in landmarks.items():
        if k in eyes_indexes:
            out_arr.append(v)
    return out_arr


def filter_upper_landmarks(landmarks):
    """Collect landmarks of upper face - eyes, upper nose, eyebrows"""
    eyes_indexes = [_ for _ in range(37, 48)]
    eyebrows_indexes = [_ for _ in range(18, 27)]
    rest_visible = [1, 17, 28]
    out_arr = []
    for (k, v) in landmarks.items():
        if k in eyes_indexes + eyebrows_indexes + rest_visible:
            out_arr.append(v)
    return out_arr


def biggest_surface(rectangles: dlib.rectangles) -> dlib.rectangle:
    """Selects rectangle with biggest area."""
    return max([fix_rect(r) for r in rectangles], key=lambda x: x.area())


def crop(image: Image, rect: dlib.rectangle) -> np.ndarray:
    """Cuts image to rectangle coordinates."""
    return image.obj[rect.top(): rect.bottom(), rect.left(): rect.right()]


def gouge_landmarks_values(face_image: NDArray[Any], coordinates):
    for y, row in enumerate(face_image):
        for x, column in enumerate(row):
            if not (x, y) in coordinates:
                face_image[y][x] = np.asarray([0, 0, 0])
    return face_image


class FaceDetector:
    """Contains detector from `dlib`."""
    landmarks_predictor = LandmarksPredictor(
        predictor_fp=None, show_samples=False,
        face_detection=True, auto_download=True)

    def __init__(self) -> None:
        self._detector = get_frontal_face_detector()
        self._real_masks = [MaskingStrategy.blue, MaskingStrategy.grey]
        self._masks = self._real_masks + [MaskingStrategy.black_box]
        self._last_mask = random.choice(self._real_masks)

    def vector_generator(
            self, df: pd.DataFrame, vector_func: Callable, landmarks_detection: bool = True
    ) -> Tuple[NDArray[Any], Image]:
        """Creates generator which yields vectors created by openface from images and this image."""

        for index, row in df.iterrows():
            img = Image(row["filename"], row["identity"])
            face_rectangles = self._detector(img.obj, 1)

            if face_rectangles:
                rect = biggest_surface(face_rectangles)
                face_crop = crop(img, rect)  # landmarks instead of whole face
                if landmarks_detection:
                    self.landmarks_predictor.detect(
                        images_list=[(face_crop, row["filename"])],
                        create_map=False
                    )
                    landmarks = self.landmarks_predictor.get_landmarks()
                    important_coords = filter_upper_landmarks(next(iter(landmarks.values())))
                    face_crop = gouge_landmarks_values(face_crop, important_coords)
                    self.landmarks_predictor.forget_landmarks()
                else:
                    if row["impose_mask"]:
                        if row["impose_mask"] == MaskingStrategy.alternately:
                            img.switch_mask_imposer_mask(self._last_mask.value)
                            self._last_mask = \
                                [x for x in self._real_masks if x != self._last_mask][0]
                        elif row["impose_mask"] in self._masks:
                            img.switch_mask_imposer_mask(row["impose_mask"])
                        face_crop = img.get_masked((face_crop, "m!_" + str(img.path)))
                yield vector_func(face_crop), img
