from enum import Enum


class MaskingStrategy(Enum):
    grey = 2
    blue = 1
    black_box = 0
    alternately = "alternately"

    def __str__(self) -> str:
        return str(self.value)
