from typing import Optional

import cv2
import imutils


# from mask_imposer import MaskImposer

class Image:
    # mask_imposer: MaskImposer = MaskImposer()
    def __init__(self, path: str, identity: Optional[str] = None, size=(96, 96)) -> None:
        self.obj = cv2.imread(path)
        self.masked_obj = None
        self.identity = identity
        self.path = path
        self._face = None
        self.target_size = size

    def get_masked(self):
        # if self.masked_obj is None:
        # self.masked_obj = self.mask_imposer.impose_mask(self.path)
        return self.masked_obj

    def get_face(self):
        return self._face

    def set_face(self, face):
        if self._face is None:
            # height, width, _ = face.shape
            # if (height,width) != self.target_size:
            self._face = imutils.resize(face, *self.target_size)

    def __repr__(self):
        return f"{self.__class__.__name__}(identity={self.identity!r}, path={self.path!r}, size={self.obj.shape!r})"
