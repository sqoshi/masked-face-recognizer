# Masked Face Recognizer

Project is a part of series related with my Bachelor of Science Thesis research.

## Table of contents

- [Introduction](#introduction)
- [General](#general)
    - [Accepted datasets structures](#accepted-datasets-structures)

## Introduction

Face recognizer in terms of machine learning is a model which takes images as an input and
recognizes who is on it.

Masked Face Recognizer is a model which recognizes person from image even with facial mask on.

Package offers' algorithm to create such a models.

## General

At the beginning program takes a path to a directory containing dataset.

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
our embeddings as an input. Model is saved in `.h5` file.

So if you have given 3 identities than result of prediction is a list of probabilities, where the
highest means `This is person has the biggest probability to be on that image.`.

Example

model has been trained on identities: 

`[kate, jacob, peter]`

output:
`[0.1, 0.3, 0.6]`

What means that the most probably peter is on input image.
