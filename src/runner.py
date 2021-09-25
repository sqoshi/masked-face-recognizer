import logging
import coloredlogs

from preanalysis.analyzer import DatasetReader
from preanalysis.defs import DatasetConfig
from predictions.extractor import FaceExtractor
from predictions.face_recognizer import FaceRecognizer
from predictions.trainers.svm_trainer import SVMTrainer

logger = logging.getLogger(__name__)
coloredlogs.install(
    filename="decoder.log",
    filemode='a',
    level="DEBUG"
)

if __name__ == '__main__':
    logger.info("Program started.")
    dfr = DatasetReader(
        # DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/original", None),
        DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/celeba/img_align_celeba",
                      "/home/piotr/Documents/bsc-thesis/datasets/celeba/identity_CelebA.txt")
    )

    dataset = dfr.read(25, equalize=True)
    train_set, test_set = dfr.split_dataset(dataset)
    print(f"{len(dataset)} == {len(train_set)} + {len(test_set)}")
    fd = FaceExtractor(train_set)
    embs = fd.extract()
    fd.save()
    # embs = "../face_vectors.pickle"

    t = SVMTrainer(embs)
    m = t.train()
    label_coder = t.label_encoder
    t.store_model()

    fr = FaceRecognizer(m, test_set, label_coder)
    fr.recognize()
    logger.info("Program ended.")
