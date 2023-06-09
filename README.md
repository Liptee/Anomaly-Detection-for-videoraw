This repository contains tools to train your own model for detecting abnormal human behaviour using the [mediapipe](https://developers.google.com/mediapipe) framework. The models are built based on autoencoders with different architectures:
* A simple Neural Network with fully-connected layers (**unrealized**)
* Based on 1D Convolutional Neural Network (**unrealized**)
* Based on 2D Convolutional Neural Network
* Based on 3D Convolutional Neural Network (**if it needs to realize**)
* Transformer Based

## Requirements
At the current moment, the **mediapipe** framework doest not work with the latest version of Python. Repository was developed on Python 3.8, so it's most recommended.

List of using libraries:
* **mediapipe** 0.10.0
* **opencv-python** 4.7.0.72
* **scikit-learn** 1.2.2
* **torch** 2.0.1
* **torchvision** 0.15.2
* **tqdm** 4.65.0
* **numpy** 1.24.3

## How to use
The main class in this code that you will need to work with is the `Trainer` class. It allows you to specify model type, model parameters, training parameters, add data for training and validation, as well as create a set of abnormal data, and of course train the model.

Initialization of the class looks like this:
```python
from trainer import Trainer
from parameters import init_transformer_params

params = init_transformer_params(init_transformer_params())
```
The `init_transformer_params` function returns a dictionary with the default parameters for the Transformer model. You can change them as you like, or create your own function to initialize the parameters.
