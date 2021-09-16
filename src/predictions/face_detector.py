import cv2
import dlib
import pandas as pd
import warnings
from _dlib_pybind11 import get_frontal_face_detector
from imutils import resize

from src.predictions.image import Image


def rect_to_bb(rect: dlib.rectangle):
    """Transform dlib rectangle to left,right cords and width, height."""
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def draw_sample(image, rect_list, crop=True, delay=1000):
    def _display_img(img, dly):
        cv2.imshow("Output Sample", img)
        cv2.waitKey(dly)

    for rect in rect_list:
        x, y, w, h = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (30, 144, 255), 2)
        cv2.putText(image, "Face", (x - 10, y - 10), cv2.FONT_ITALIC, 0.5, (90, 158, 233), 2)
        if crop:
            image = image[rect.top():rect.bottom(), rect.left():rect.right()]
            image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
            _display_img(image, delay)

    if not crop:
        _display_img(image, delay)


def warn_detections(face_detections):
    if len(face_detections) > 1:
        warnings.warn(f"Detected {len(face_detections)} faces on image.")
    elif len(face_detections) == 0:
        warnings.warn("Could not detect face on image.")


def crop(image, rect):
    return image.obj[rect.top():rect.bottom(), rect.left():rect.right()]


class FaceDetector:
    def __init__(self, dataset_df: pd.DataFrame, embedder_fp="../models/nn4.small2.v1.t7") -> None:
        self.dataset_df = dataset_df
        self._detector = get_frontal_face_detector()
        self._embedder = cv2.dnn.readNetFromTorch(embedder_fp)
        self.target_size = (96, 96)

    def vector(self, face_crop):
        face_blob = cv2.dnn.blobFromImage(face_crop, 1.0 / 255, self.target_size, (0, 0, 0), swapRB=True, crop=False)
        self._embedder.setInput(face_blob)
        vec = self._embedder.forward()
        print(f"Face vector shape is {vec.shape}.")
        return vec.flatten()

    def detect(self):
        for index, row in self.dataset_df.iterrows():
            img = Image(row['filename'], row['identity'])
            print(img)
            face_rectangles = self._detector(img.obj, 1)
            warn_detections(face_rectangles)
            # draw_sample(img.obj, face_rectangles, True)
            # todo: adjust rectangle
            rect = face_rectangles.pop()
            face_crop = crop(img.obj, rect)
            # img.set_face(face_crop)
            embeddings_vec = self.vector(face_crop)
            print(embeddings_vec)
