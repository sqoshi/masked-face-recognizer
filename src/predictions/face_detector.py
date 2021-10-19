from _dlib_pybind11 import get_frontal_face_detector


class FaceDetector:
    """Contains detector from `dlib`."""

    def __init__(self) -> None:
        self._detector = get_frontal_face_detector()
