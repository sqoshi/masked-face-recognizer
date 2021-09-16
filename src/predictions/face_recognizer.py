import pandas as pd

from src.predictions.embedder import Embedder
from src.predictions.extractor import warn_detections, biggest_surface, crop
from src.predictions.face_detector import FaceDetector
from src.predictions.image import Image


class FaceRecognizer(FaceDetector, Embedder):
    def __init__(self, model, test_set: pd.DataFrame):
        FaceDetector.__init__(self)
        Embedder.__init__(self)
        self._model = model
        self.test_set_df = test_set
        self._standard_width = 600

    def recognize(self):
        for index, row in self.test_set_df.iterrows():
            img = Image(row['filename'], row['identity'])
            face_rectangles = self._detector(img.obj, 1)
            warn_detections(face_rectangles)
            if not face_rectangles:
                continue
            rect = biggest_surface(face_rectangles)
            face_crop = crop(img, rect)
            embeddings_vec = self.vector(face_crop)
            preds = self._model.predict_proba(embeddings_vec)[0]
            print(preds)
