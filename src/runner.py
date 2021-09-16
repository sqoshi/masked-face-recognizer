from preanalysis.analyzer import DatasetReader
from src.preanalysis.dataclasses import DatasetConfig
from src.predictions.extractor import FaceExtractor
from src.predictions.face_recognizer import FaceRecognizer
from src.predictions.trainers.svm_trainer import SVMTrainer

if __name__ == '__main__':
    # TODO: prepare table with statistics -> top1 - > acc  + top5 - > acc
    da = DatasetReader(
        DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/original", None),
        # DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/celeba/img_align_celeba",
        #               "/home/piotr/Documents/bsc-thesis/datasets/celeba/identity_CelebA.txt")
    )
    dataset_df = da.read(8, equalize=True)
    da2 = DatasetReader(
        DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/original", None),
        # DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/celeba/img_align_celeba",
        #               "/home/piotr/Documents/bsc-thesis/datasets/celeba/identity_CelebA.txt")
    )
    train_set = da2.read(1, equalize=True)

    # fd = FaceExtractor(dataset_df)
    # embs = fd.extract()
    # fd.save()
    embs = "../face_vectors.pickle"
    t = SVMTrainer(embs)
    m = t.train()
    label_coder = t.label_encoder
    t.store_model()

    fr = FaceRecognizer(m, train_set, label_coder)
    fr.recognize()
