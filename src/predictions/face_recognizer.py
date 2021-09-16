import numpy as np
import pandas as pd
from termcolor import cprint

from src.predictions.embedder import Embedder
from src.predictions.extractor import warn_detections, biggest_surface, crop
from src.predictions.face_detector import FaceDetector
from src.predictions.image import Image


class FaceRecognizer(FaceDetector, Embedder):
    def __init__(self, model, test_set: pd.DataFrame, label_encoder):
        FaceDetector.__init__(self)
        Embedder.__init__(self)
        self._model = model
        self.test_set_df = test_set
        self._standard_width = 600
        self.label_encoder = label_encoder

    def print_stats(self, predictions, correct_identity, stats):
        arr = np.array(predictions)
        top5_indexes = arr.argsort()[-5:][::-1]
        # print(top5_indexes)
        # proba = preds[j]
        classes = [self.label_encoder.classes_[j] for j in top5_indexes]
        probs = [predictions[idx] for idx in top5_indexes]
        print(classes)
        print(probs)
        if correct_identity in classes:
            cprint(f"Correct id {correct_identity} is in top5.", "yellow")
            if probs[classes.index(correct_identity)] == max(probs):
                cprint(f"Perfect match id {correct_identity} is has best prob.", "green")
                stats[0] = stats[0] + 1
            else:
                stats[1] = stats[1] + 1
        else:
            cprint(f"Wrong match {correct_identity} person not in top5.", "red")
            stats[2] = stats[2] + 1

    def recognize(self):
        stats = [0, 0, 0]
        for index, row in self.test_set_df.iterrows():
            img = Image(row['filename'], row['identity'])
            face_rectangles = self._detector(img.obj, 1)
            # warn_detections(face_rectangles)
            if not face_rectangles:
                continue
            rect = biggest_surface(face_rectangles)
            face_crop = crop(img, rect)
            embeddings_vec = self.vector(face_crop)
            preds = self._model.predict_proba(embeddings_vec)[0]
            self.print_stats(preds, img.identity, stats)
        cprint(f"{stats[0]} perfects.", "green")
        cprint(f"{stats[1]} top5.", "yellow")
        cprint(f"{stats[2]} fails.", "red")
