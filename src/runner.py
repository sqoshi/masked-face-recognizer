from preanalysis.analyzer import DatasetReader
from src.preanalysis.dataclasses import DatasetConfig

if __name__ == '__main__':
    da = DatasetReader(
        DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/original", None),
        # DatasetConfig("/home/piotr/Documents/bsc-thesis/datasets/celeba/img_align_celeba",
        #               "/home/piotr/Documents/bsc-thesis/datasets/celeba/identity_CelebA.txt")
    )
    da.read(2, equalize=True)
