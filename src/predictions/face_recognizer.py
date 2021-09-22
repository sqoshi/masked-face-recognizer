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

    def update_stats(self, predictions, correct_identity, stats):
        arr = np.array(predictions)
        top5_indexes = arr.argsort()[-5:][::-1]
        # print(top5_indexes)
        # proba = preds[j]
        classes = [self.label_encoder.classes_[j] for j in top5_indexes]
        probs = [predictions[idx] for idx in top5_indexes]
        # print(classes)
        # print(probs)
        if correct_identity in classes:
            if probs[classes.index(correct_identity)] == max(probs):
                cprint(f"Perfect match id {correct_identity} has best prob.", "green")
                stats["perfect"] = stats["perfect"] + 1
            else:
                cprint(f"Correct id {correct_identity} is in top5.", "yellow")
                stats["top5"] = stats["top5"] + 1
        else:
            cprint(f"Wrong match {correct_identity} person not in top5.", "red")
            stats["fail"] = stats["fail"] + 1

    def print_stats(self, stats):
        cprint(f"{stats['perfect']} perfects.", "green")
        cprint(f"{stats['top5']} top5.", "yellow")
        cprint(f"{stats['fail']} fails.", "red")
        tests_number = sum(stats.values())
        opt = round((stats['perfect'] + stats['top5']) / tests_number * 100, 3)
        pes = round((stats['perfect']) / tests_number * 100, 3)

        cprint(f"Model accuracy [perfect]: {pes}", "green")
        cprint(f"Model accuracy [top5+perfect]: {opt}", "green")

    def recognize(self):
        stats = {x: 0 for x in ["perfect", "top5", "fail"]}
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
            self.update_stats(preds, img.identity, stats)
        self.print_stats(stats)
