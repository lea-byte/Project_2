# Machine Learning course EPFL - Project II -  Road Segmentation
## Description
The main goal of this project is to propose an algorithm that segments satellite images, more explicitly the algorithm will have to detect road portions on the image. We have at our disposal a set of train of 100 images, that is to say for each of the images an associated image in black and white, where the white will represent the portions of road detected and the black the rest. In short, each pixel of the images must be binary classified (road=1, background=0). We must then evaluate our model on a test set of 50 images.

All data are available via [AIcrowd page](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

The best result obtained on AIcrowd:

`Best F1 score`:
* ID : 169531
* Username : toni
* F1 score : 87.0 %
* Accuracy : 93.0 %

`Best accuracy`:
* ID : 169512
* Username : toni
* F1 score : 86.9 %
* Accuracy : 93.2 %


## Reproduce our results

## Team members
* Thomas Hasler
* Léa Grandoni
* Blerton Rashiti

## External libraries
We used the following libraries:
* [scikit-learn](https://scikit-learn.org/stable/)
* [keras](https://keras.io/)
* [tensorflow](https://www.tensorflow.org/install/)
* [numpy](https://numpy.org/)

## Python code

We have created several python files that allow us to achieve our goal:

`Data.py`

`Evaluation.py`

`Metrics.py`

`Model.py`

`Train.py`


