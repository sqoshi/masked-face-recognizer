from typing import Optional

import cv2
import numpy as np

from mask_imposer import MaskImposer


def rgba2rgb(rgba):
    print(f"rgba={rgba.shape}")
    print(f"rgba={type(rgba)}")

    row, col, ch = rgba.shape

    if ch == 3:
        return rgba
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    rgb[:, :, 0] = r * a + (1.0 - a) * 255
    rgb[:, :, 1] = g * a + (1.0 - a) * 255
    rgb[:, :, 2] = b * a + (1.0 - a) * 255
    return np.asarray(rgb, dtype='uint8')


class Image:
    mask_imposer: MaskImposer = MaskImposer()

    def __init__(self, path: str, identity: Optional[str] = None, size=(96, 96)) -> None:
        self.obj = cv2.imread(path)
        self.identity = identity
        self.path = path
        self.target_size = size
        self._face = None
        self.masked_obj = None

    def get_masked(self, optional_img=None):
        if optional_img is not None:
            return rgba2rgb(
                self.mask_imposer.impose_mask(
                    optional_img,
                    show=True
                )
            )

        if self.masked_obj is None:
            self.masked_obj = rgba2rgb(self.mask_imposer.impose_mask(self.path)[0])
            cv2.imshow("MaskedSample", self.masked_obj)
            cv2.waitKey(0)
        return self.masked_obj

    def __repr__(self):
        return f"{self.__class__.__name__}(identity={self.identity!r}," \
               f" path={self.path!r}," \
               f" size={self.obj.shape!r})"
