from collections import namedtuple
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from mask_imposer import MaskImposer
from numpy.typing import NDArray

FakeImage = namedtuple("FakeImage", "obj name")


def rgba2rgb(rgba: NDArray[Any]) -> NDArray[Any]:
    """Converts rgba image to rgb."""
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype="float32") / 255.0
    rgb[:, :, 0] = r * a + (1.0 - a) * 255
    rgb[:, :, 1] = g * a + (1.0 - a) * 255
    rgb[:, :, 2] = b * a + (1.0 - a) * 255
    return np.asarray(rgb, dtype="uint8")


class Image:
    """Class representing image and its properties.
    Gives possibility to mask image with imposer.
    """

    mask_index = 1
    mask_imposer: MaskImposer = MaskImposer(mask_index)

    def __init__(
        self,
        path: str,
        identity: Optional[str] = None,
        size: Tuple[int, int] = (96, 96),
    ) -> None:
        self.obj = cv2.imread(path)
        self.identity = identity
        self.path = path
        self.target_size = size
        self._face = None
        self.masked_obj = None

    def switch_mask_imposer_mask(self, bundled_mask_set_index: int) -> None:
        """Switches bundled mask image in mask imposer to index."""
        # mask imposer get index function is required to opt usage.
        if self.mask_index != bundled_mask_set_index:
            self.mask_imposer.switch_mask(bundled_mask_set_index)

    def get_masked(self, optional_img: Optional[Any] = None) -> NDArray[Any]:
        """Masks image with mask imposer."""
        if optional_img is not None:
            if isinstance(optional_img, tuple):
                optional_img = FakeImage(*optional_img)
            return rgba2rgb(self.mask_imposer.impose_mask(optional_img))

        if self.masked_obj is None:
            self.masked_obj = rgba2rgb(self.mask_imposer.impose_mask(self.path)[0])
            cv2.imshow("MaskedSample", self.masked_obj)
            cv2.waitKey(0)

        return self.masked_obj

    def __repr__(self) -> str:
        """String representation of image. (path is unique)"""
        return (
            f"{self.__class__.__name__}(identity={self.identity!r},"
            f" path={self.path!r},"
            f" size={self.obj.shape!r})"
        )
