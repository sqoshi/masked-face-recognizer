import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import dlib
import numpy as np
from _dlib_pybind11 import (full_object_detection, get_frontal_face_detector,
                            shape_predictor)
from numpy.typing import NDArray

from .download import download_predictor, find_predictor
from .image import Image

logger = logging.getLogger(__name__)


def rect_to_bb(rect: dlib.rectangle) -> Tuple[Any, Any, Any, Any]:
    """Transform dlib rectangle to left,right cords and width, height."""
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def shape_to_np(shape: full_object_detection, dtype: str = "int") -> NDArray:  # type:ignore
    """Transform shape object to array of landmarks cords."""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def _shape_to_dict(shape: full_object_detection) -> Dict[int, Tuple[int, int]]:
    """Transform shape object to dictionary of landmarks cords."""
    result = {}
    for i in range(0, 68):
        result[i] = (shape.part(i).x, shape.part(i).y)
    if not result:
        raise NotImplementedError()  # raise landmarks not found
    return result


def get_predictor(predictor_fp: Optional[str], auto_download: bool) -> str:
    """Looking for predictor in cwd or downloads it from dlib official page."""
    if predictor_fp is None:
        predictor_fp = find_predictor("shape_predictor_68_face_landmarks.dat")
        if predictor_fp is None:
            predictor_fp = download_predictor(
                auto=auto_download,
                predictor_name="shape_predictor_68_face_landmarks.bz2"
            )
    return predictor_fp


class LandmarksPredictor:
    def __init__(self,  # pylint:disable=R0913
                 predictor_fp: Optional[str],
                 face_detection: bool,
                 show_samples: bool,
                 auto_download: bool) -> None:
        self._detector = get_frontal_face_detector()
        self._predictor = shape_predictor(get_predictor(predictor_fp, auto_download))
        self._landmarks_collection: Dict[str, Dict[int, Tuple[int, int]]] = {}

        self._should_detect_face_rect = face_detection
        self._should_display_samples = show_samples

        self.fake_map = {}

    def forget_landmarks(self) -> None:
        self._landmarks_collection = {}
        self.fake_map = {}

    @classmethod
    def _display_sample(cls, image: Image, rect: dlib.rectangle,
                        shape: full_object_detection) -> None:
        """Display every image with drawn landmarks.

         Waiting for any key press to move to next prediction.
         """
        x, y, w, h = rect_to_bb(rect)
        cv2.rectangle(image.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image.img, "Face", (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
            2
        )
        for (x, y) in shape_to_np(shape):
            cv2.circle(image.img, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow("Output", image.img)
        cv2.waitKey(0)

    def _detect_face_rect(self, image: Image) -> dlib.rectangle:
        """Find the coordinates of rectangle in which face occurs."""

        if self._should_detect_face_rect:
            rects = self._detector(image.get_gray_img(), 1)
            if not rects:
                logger.warning(
                    "Face rectangle not found. Treating whole image as face box. "
                    "Trick is working only if image is only a front face "
                    "(see examples in readme).")
                return image.get_rectangle()
            if len(rects) == 1:
                return rects.pop()
            logger.warning(
                f"Detected multiple faces on image `{image}`. Taking first found."
            )
            return rects.pop()
        # whole image as face rectangle (there should be only a center face on image)
        return image.get_rectangle()

    def _check_fails(self, images_list: List[str]) -> None:
        """Check if landmarks were found for every inputted image and warns about fails."""
        if len(images_list) != len(self._landmarks_collection):
            diff = len(images_list) - len(self._landmarks_collection)
            logger.warning(f"Landmarks not found in {diff} images.")

    def detect(self, images_list: List[Union[str, tuple]], create_map: bool) -> None:
        """Creates landmark collection.

        During creation may optionally display samples with drawn landmarks.
        May detect face boxes, but it is preferred to pass images as stated in readme.
        """
        for img_path in images_list:
            image = Image(img_path)
            try:
                rect = self._detect_face_rect(image)  # detect rectangles with faces

                shape = self._predictor(image.get_gray_img(), rect)  # detect landmarks

                self._landmarks_collection[str(image)] = _shape_to_dict(shape)

                if create_map:
                    self.fake_map[str(image)] = image.img

                if self._should_display_samples:
                    self._display_sample(image, rect, shape)

            except NotImplementedError:  # must be changed
                logger.warning(f"Landmarks not detected on {image}.")
                continue
        self._check_fails(images_list)
        logger.info("Detection finished.")

    def get_landmarks(self) -> Dict[str, Dict[int, Tuple[int, int]]]:
        return self._landmarks_collection

