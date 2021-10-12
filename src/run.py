import logging
import coloredlogs

from preanalysis.dataset_config import DatasetConfigBuilder
from preanalysis.reader import DatasetReader
from predictions.extractor import FaceExtractor
from predictions.face_recognizer import FaceRecognizer
from predictions.trainers.svm_trainer import SVMTrainer

logging.basicConfig(filename="masked_face_recognizer.log")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")


def main():
    logger.info("Program started.")
    logger.info("1. Dataset reading stage.")
    dsc_builder = DatasetConfigBuilder()
    dr = DatasetReader(
        dsc_builder.build("/home/piotr/Documents/bsc-thesis/datasets/original")
        # dsc_builder.build("/home/piotr/Documents/bsc-thesis/datasets/celeba")
    )
    dataset = dr.read()
    train_set, test_set = dr.split_dataset(dataset)

    logger.info("2. Extracting embeddings stage.")
    fd = FaceExtractor(train_set)
    embs = fd.extract() ; fd.save()
    # embs = "../face_vectors.pickle"

    logger.info("3. Model training stage")
    t = SVMTrainer(embs)
    m = t.train()
    label_coder = t.label_encoder
    t.store_model()

    logger.info("4. Face recognition stage.")
    fr = FaceRecognizer(m, test_set, label_coder)
    fr.recognize()
    logger.info("Program ended.")


if __name__ == '__main__':
    main()
