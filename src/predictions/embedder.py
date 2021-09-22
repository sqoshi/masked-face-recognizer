import traceback

import cv2
from termcolor import cprint


class Embedder:
    def __init__(self, embedder_fp="../models/nn4.small2.v1.t7", input_shape=(96, 96)):
        self._embedder = cv2.dnn.readNetFromTorch(embedder_fp)
        self._embedder_input_shape = input_shape

    def vector(self, face_crop):
        try:
            face_blob = cv2.dnn.blobFromImage(face_crop, 1.0 / 255, self._embedder_input_shape, (0, 0, 0), swapRB=True,
                                              crop=False)
            self._embedder.setInput(face_blob)
            vec = self._embedder.forward()
            # cprint(f"Face vector shape is {vec.shape}.", "yellow")
            return vec
        except:
            print(face_crop.shape)
            cv2.imshow("Error image", face_crop)
            traceback.print_exc()
