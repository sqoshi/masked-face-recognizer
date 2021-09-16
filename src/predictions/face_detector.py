from _dlib_pybind11 import get_frontal_face_detector

from src.predictions.image import Image


class FaceDetector:
    def __init__(self):
        self._detector = get_frontal_face_detector()

    # def detect(self, img: Image):
    #     face_rectangles = self._detector(img.obj, 1)
    #     warn_detections(face_rectangles)
    #     if not face_rectangles:
    #         continue
    #     rect = biggest_surface(face_rectangles)
