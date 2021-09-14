from typing import Optional

import cv2


class Image:
    def __init__(self, path: str, identity: Optional[str] = None) -> None:
        self.obj = cv2.imread(path)
        self.identity = identity
        self.path = path

    def __repr__(self):
        return f"{self.__class__.__name__}(identity={self.identity!r}, path={self.path!r}, size={self.obj.shape!r})"
