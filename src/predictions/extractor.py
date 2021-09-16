import pickle

import cv2
import dlib
import imutils
import numpy as np
import pandas as pd
import warnings
from _dlib_pybind11 import get_frontal_face_detector
from termcolor import cprint

from src.predictions.face_detector import FaceDetector
from src.predictions.image import Image


def rect_to_bb(rect: dlib.rectangle):
    """Transform dlib rectangle to left,right cords and width, height."""
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def biggest_surface(rectangles: dlib.rectangles) -> dlib.rectangle:
    # cprint("Selecting rectangle with biggest surface.", "yellow")
    surface = 0
    biggest_rect = None
    for rect in rectangles:
        x, y, w, h = rect_to_bb(rect)
        if w * h > surface:
            biggest_rect = rect
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
        warnings.warn(f"Detected {len(face_detections)} faces on image.")
    elif len(face_detections) == 0:
        warnings.warn("Could not detect face on image.")


def crop(image: Image, rect: dlib.rectangle) -> np.ndarray:
    return image.obj[rect.top():rect.bottom(), rect.left():rect.right()]


class FaceExtractor(FaceDetector):
    def __init__(self, dataset_df: pd.DataFrame, embedder_fp="../models/nn4.small2.v1.t7", face_size=(96, 96)) -> None:
        super().__init__()
        self.dataset_df = dataset_df
        self._embedder = cv2.dnn.readNetFromTorch(embedder_fp)
        self.target_size = face_size
        self._embeddings = {"vectors": [], "classes": []}

    def save(self, fn="../face_vectors.pickle"):
        with open(fn, 'wb') as fw:
            pickle.dump(self._embeddings, fw, protocol=pickle.HIGHEST_PROTOCOL)

    def vector(self, face_crop):
        face_blob = cv2.dnn.blobFromImage(face_crop, 1.0 / 255, self.target_size, (0, 0, 0), swapRB=True, crop=False)
        self._embedder.setInput(face_blob)
        vec = self._embedder.forward()
        cprint(f"Face vector shape is {vec.shape}.", "yellow")
        return vec.flatten()

    def extract(self):
        for index, row in self.dataset_df.iterrows():
            img = Image(row['filename'], row['identity'])
            cprint(img, "green")
            face_rectangles = self._detector(img.obj, 1)
            warn_detections(face_rectangles)
            # draw_sample(img.obj, face_rectangles, True)
            # todo: adjust rectangle
            if not face_rectangles:
                continue
            rect = biggest_surface(face_rectangles)
            face_crop = crop(img, rect)
            embeddings_vec = self.vector(face_crop)
            self._embeddings["vectors"].append(embeddings_vec)
            self._embeddings["classes"].append(img.identity)
        return self._embeddings