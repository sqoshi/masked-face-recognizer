# Masked Face Recognizer

Project is a part of series related with my Bachelor of Science Thesis research.

## Table of contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
    - [Configuration](#configuration)
- [General](#general)
    - [Accepted datasets structures](#accepted-datasets-structures)
    - [Openface](#openface)
    - [Train phase](#train-phase)
- [Code Example](#code-example)
- [Results visualization](#results-visualization)

## Introduction

Face recognizer in terms of machine learning is a model which takes images as an input and
recognizes person who is in it. This program requires only clean dataset with frontal faces as
CelebA, the dataset is dynamically modified accordingly with input Configuration, has also the
possibility to perform multiple learning process and create multiple models. Process is fully
automated and to run program we need to specify the path to dataset and configuration or series of
configurations which program should perform in `src/main.py`.

## Installation

```shell
pip3 install -r requirements.txt
```

## Usage

Program requires two things to run.

1. User need to manually find a dataset with frontal faces and download it on local machine.
2. User need to configure which modifications such as dataset reading limitations, learning
   procedure modifications and used model structure.

To achieve above requirements' user need to pass path to a dataset in `src/main.py` and configure
in which way should model be learned.

### Configuration

To configure user need to push configuration such as in below example. Default config is a config
without modification.

```python
configurator = Configurator(datasets[-1])  # pass path to your dataset here
configurator.push(configurator.default())  # push config to a configurator, examples in class
```

Configuration examples are available in Configurator docstrings, push maybe also achieved like
that:

```python
configurator.push(
    Configuration(
        name=f"{int(p * 100)}p_masked_train_set",
        dataset_path=self.dataset_path,
        split_ratio=0.6,
        personal_images_quantity=30,
        identities_limit=50,
        equal_piqs=True,
        train_set_modifications=DatasetModifications(
            mask_ratio=p, inplace=True, mask=MaskingStrategy.alternately
        ),
        test_set_modifications=DatasetModifications(
            mask_ratio=1.0, inplace=True, mask=MaskingStrategy.alternately
        ),
    )
)
```

Configuration arguments description:

```python
dataset_path: str,  # path to a dataset
split_ratio: float,  # ratio in which dataset will be split into a train and test set
personal_images_quantity: int = 1,  # reading limitation - that a number of photos per person read
equal_piqs: bool = False,  # if true all people will have exactly personal_images_quantity per person
skip_unknown: bool = False,  # skip personalities without unknown identity
identities_limit: Optional[int] = None,  # read only identities_limit identities from database
landmarks_detection: bool = True,  # create embeddings vector from visible landmarks instead of whole image
test_set_modifications: Optional[DatasetModifications] = None,
train_set_modifications: Optional[DatasetModifications] = None,
name: Optional[str] = None,  # set configuration name such as default
svm_config: SVMStructure = None,  # model structure as used kernel and other parameters 
```

SVM available parametrs:

```python
 SVMStructure = {
    "name": "sklearn.svm.SVC",
    "details": {
        "C": self._model.C,
        "kernel": self._model.kernel,
        "probability": self._model.probability,
        "degree": self._model.degree,
        "gamma": self._model.gamma,
        "coef0": self._model.coef0,
        "tol": self._model.tol,
        "shrinking": self._model.shrinking,
        "class_weight": self._model.class_weight,
        "verbose": self._model.verbose,
        "max_iter": self._model.max_iter,
        "decision_function_shape": self._model.decision_function_shape,
        "break_ties": self._model.break_ties,
        "random_state": self._model.random_state,
    },
}
```

Dataset available modifications:

```python
DatasetModifications(
    mask_ratio=0.7,  # if true then 70% of images from set will be masked
    inplace=False,
    # if false then images in set will be masked, else set will be extended by copies of masked images
    mask=MaskingStrategy.alternately
    # mask images with all mask available from mask_imposer alternately
)
```

## General

### Accepted datasets structures

Datasets maybe kept in two different structures:

1. Grouped - where subdirectories name determine identity on images inside.

```
grouped_dataset_root_directory/
│
├── person1_dir/
│      ├── img1.png
│      ├── img2.png
│      └── img3.png
│
├── person2_dir/
│      ├── img1.png
│      ├── img2.png
│      └── img3.png
│
└── description.(md|txt) [OPTIONAL]
```

2. Mixed - where identities must be determined in `identities.(csv|txt)` file.

```
mixed_dataset_root_directory/
│
├── images_directory/
│      ├── img1.png
│      ├── img2.png
│      ├── img3.png
│      └── img4.png
│
├── identities.(csv|txt)
│
└── description.(md|txt) [OPTIONAL]
```

Identities file must contain minimally two columns with image name in first column and identity in
second.

```yaml
000001.jpg jacob
000002.jpg kate
000003.jpg jacob
000004.jpg peter
```

*_Filename not absolute file path!._*

### [Openface]()

Dataset is being split with given ratio to train and test sets.

When dataset has been read program iterates over all images, detects rectangle with face inside,
crops that part and using openface `nn4.small2.v1.t7`
model transforming images to 128-D flat vectors (embeddings).

### Train phase

Model is trained on train set. For now program allow use only `sklearn.svm.SVC` model which takes
our embeddings as an input. Model is saved in `.pickle` file.

So if you have given 3 identities than result of prediction is a list of probabilities, where the
highest means `This is person has the biggest probability to be on that image.`.

Example

model has been trained on identities:

`[kate, jacob, peter]`

output:
`[0.1, 0.3, 0.6]`

What means that the most probably peter is on input image.

## Code Example

```python
    def extract(
        self, df: pd.DataFrame, landmarks_detection: bool
) -> Dict[str, List[NDArray[Any]]]:
    """Extracting embeddings from loaded images."""
    logger.info("Extracting embeddings.")
    for i, (vec, img) in enumerate(
            self.vector_generator(df, self.vector, landmarks_detection)
    ):
        logger.info("Extracting (%s/%s) ...", i, len(df.index))
        embeddings_vec = vec.flatten()
        self._upload_embeddings(img.identity, embeddings_vec)

    return self._embeddings
```

## Results visualization

Results may be visualized by [MFR-viz](https://github.com/sqoshi/masked-face-recognizer-frontend).

Before launching MFR-viz, please deploy results with ```python3 deploy.sh```.