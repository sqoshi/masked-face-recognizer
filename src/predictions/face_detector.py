import cv2
import dlib
import pandas as pd
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
    for rect in rect_list:
        x, y, w, h = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (30, 144, 255), 2)
        cv2.putText(image, "Face", (x - 10, y - 10), cv2.FONT_ITALIC, 0.5, (90, 158, 233), 2)

        if crop:
            image = image[rect.top():rect.bottom(), rect.left():rect.right()]
            image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
            cv2.imshow("Output Sample", image)
            cv2.waitKey(delay)
    if not crop:
        cv2.imshow("Output Sample", image)
        cv2.waitKey(delay)


class FaceDetector:
    def __init__(self, dataset_df: pd.DataFrame) -> None:
        self.dataset_df = dataset_df
        self._detector = get_frontal_face_detector()

    def detect(self):
        for index, row in self.dataset_df.iterrows():
            img = Image(row['filename'], row['identity'])
            print(img)
            face_rectangles = self._detector(img.obj, 1)
            draw_sample(img.obj, face_rectangles, True)
