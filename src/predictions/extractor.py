import logging
import pickle

import cv2
import dlib
import imutils
import numpy as np
import pandas as pd
import warnings
from termcolor import cprint

from predictions.embedder import Embedder
from predictions.face_detector import FaceDetector
from predictions.image import Image

logger = logging.getLogger(__name__)


def rect_to_bb(rect: dlib.rectangle):
    """Transform dlib rectangle to left,right cords and width, height."""
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def fix_rect(rect: dlib.rectangle):
    return dlib.rectangle(
        top=nn(rect.top()),
        bottom=nn(rect.bottom()),
        left=nn(rect.left()),
        right=nn(rect.right())
    )


def biggest_surface(rectangles: dlib.rectangles) -> dlib.rectangle:
    logger.info("Selecting face rectangle with biggest area.")
    biggest_surf = 0
    biggest_rect = None
    for rect in rectangles:
        rect = fix_rect(rect)
        if rect.area() > biggest_surf:
            biggest_rect = rect
            biggest_surf = rect.area()
    return biggest_rect


def draw_sample(image, rect_list, crop_face=False, delay=1):
    # something working wrong
    def _display_img(img, dly):
        cv2.imshow("Output Sample", img)
        cv2.waitKey(dly)

    for rect in rect_list:
        x, y, w, h = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (30, 144, 255), 2)
        cv2.putText(image, "Face", (x - 10, y - 10), cv2.FONT_ITALIC, 0.5, (90, 158, 233), 2)

        if crop_face:
            image = image[rect.top():rect.bottom(), rect.left():rect.right()]
            image = imutils.resize(image, 96, 96)
            _display_img(image, delay)

    if not crop_face:
        _display_img(image, delay)


def warn_detections(face_detections: dlib.rectangles) -> None:
    if len(face_detections) > 1:
        logger.warning(f"Detected {len(face_detections)} faces on image. The biggest surface face will be processed.")
    elif len(face_detections) == 0:
        logger.warning("Could not detect face on image.")


def nn(value: int) -> int:
    """cast value to closest non negative value"""
    return 0 if value < 0 else value


def crop(image: Image, rect: dlib.rectangle) -> np.ndarray:
    return image.obj[rect.top():rect.bottom(), rect.left():rect.right()]


class FaceExtractor(FaceDetector, Embedder):
    def __init__(self, dataset_df: pd.DataFrame) -> None:
        FaceDetector.__init__(self)  # explicit calls without super
        Embedder.__init__(self)
        self.dataset_df = dataset_df
        self._embeddings = {"vectors": [], "classes": []}

    def save(self, fn="../face_vectors.pickle"):
        logger.info("Saving embeddings to %s." % fn)
        with open(fn, 'wb') as fw:
            pickle.dump(self._embeddings, fw, protocol=pickle.HIGHEST_PROTOCOL)

    def extract(self):
        logger.info("Extracting embeddings")
        for index, row in self.dataset_df.reset_index(drop=True).iterrows():
            logger.info(f"Extracting (%s/%s) ...", index, len(self.dataset_df))
            img = Image(row['filename'], row['identity'])
            face_rectangles = self._detector(img.obj, 1)
            warn_detections(face_rectangles)
            # todo: adjust rectangle
            if not face_rectangles:
                continue
            rect = biggest_surface(face_rectangles)
            face_crop = crop(img, rect)
            embeddings_vec = self.vector(face_crop).flatten()
            self._embeddings["vectors"].append(embeddings_vec)
            self._embeddings["classes"].append(img.identity)
        return self._embeddings
