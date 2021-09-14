from preanalysis.analyzer import DatasetReader
from src.preanalysis.dataclasses import DatasetConfig
from src.predictions.face_detector import FaceDetector

if __name__ == '__main__':
    # TODO: prepare table with statistics -> top1 - > acc  + top5 - > acc
    da = DatasetReader(
        DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/original", None),
        # DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/celeba/img_align_celeba",
        #               "/home/piotr/Documents/bsc-thesis/datasets/celeba/identity_CelebA.txt")
    )
    dataset_df = da.read(25, equalize=True)

    fd = FaceDetector(dataset_df)
    fd.detect()
