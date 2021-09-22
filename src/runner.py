from preanalysis.analyzer import DatasetReader
from src.preanalysis.defs import DatasetConfig
from src.predictions.extractor import FaceExtractor
from src.predictions.face_recognizer import FaceRecognizer
from src.predictions.trainers.svm_trainer import SVMTrainer

if __name__ == '__main__':
    # TODO: prepare table with statistics -> top1 - > acc  + top5 - > acc
    # da = DatasetReader(
    #     DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/original", None),
    #     # DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/celeba/img_align_celeba",
    #     #               "/home/piotr/Documents/bsc-thesis/datasets/celeba/identity_CelebA.txt")
    # )
    # dataset_df = da.read(8, equalize=True)
    dfr = DatasetReader(
        DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/original", None),
        # DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/celeba/img_align_celeba",
        #               "/home/piotr/Documents/bsc-thesis/datasets/celeba/identity_CelebA.txt")
    )
    dataset = dfr.read(equalize=True)
    train_set, test_set = dfr.split_dataset(dataset)

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
