from typing import Optional, Union, Any, Tuple

from _dlib_pybind11 import rectangle
from cv2 import COLOR_BGR2BGRA, COLOR_BGR2GRAY, cvtColor, imread
from numpy.typing import NDArray


def set_img(file: Union[str, Tuple[NDArray[Any], str]]) -> Tuple[NDArray[Any], str]:
    if isinstance(file, tuple):
        fake_path = [f for f in file if isinstance(f, str)].pop()
        np_arr = [f for f in file if not isinstance(f, str)].pop()
        return np_arr, fake_path
    return imread(file, -1), file


class Image:
    """Hold image data."""

    def __init__(self, file: Union[str, Tuple[str, NDArray[Any]]]) -> None:
        self.img, self.__name = set_img(file)
        self._gray_img: Optional[cvtColor] = None
        self._rect: Optional[rectangle] = None
        if self.img.shape[-1] == 3:
            self.img = self.converted_rgba()

    def __str__(self) -> str:
        return self.__name

    def get_gray_img(self) -> cvtColor:
        """Creates if not yet created image in gray scale."""
        if self._gray_img is None:
            self._gray_img = cvtColor(self.img, COLOR_BGR2GRAY)
        return self._gray_img

    def get_rectangle(self) -> rectangle:
        """Creates if not yet created dlib rectangle of within whole image."""
        if self._rect is None:
            height, width, _ = self.img.shape
            self._rect = rectangle(left=0, top=0, right=width, bottom=height)
        return self._rect

    def converted_rgba(self) -> cvtColor:
        """Converts 3 channel image to 4 channel."""
        return cvtColor(self.img, COLOR_BGR2BGRA)
