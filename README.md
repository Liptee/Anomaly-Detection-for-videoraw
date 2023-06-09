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


### Initialization of the Trainer
Initialization of the class looks like this:
```python
from trainer import Trainer
from parameters import init_transformer_params

model = init_transformer_params(init_transformer_params())
```
The `init_transformer_params` function returns a dictionary with the default parameters for the Transformer model. You can change them as you like, or create your own function to initialize the parameters.

You can also specify the following parameters during initialization:
* `params` - dictionary with parameters for the model. From `parameters` module you can import functions to initialize parameters for different models.
* `model`: str - model type. Can be one of the following: `nn`, `cnn`, `transformer`. Default: `transformer`. 
* `criterin` - loss function. Default: `MSELoss()`.
* `learning_rate` - learning rate. Default: `0.001`.
* `optimizer` - optimizer. Default: `Adam`.
* `device` - device for training. Default: `cpu`.
* `batch_size` - batch size. Default: `32`.
* `num_epochs` - number of epochs. Default: `10`.
* `sequence_length` - length of the sequence. Default: `12`.

Note that when you initialize parameters from `parameters`, you can also specify different values for model parameters.

### Adding data
For adding data you can use several methods: `add_data`, `add_validation_data`, `add_abnormal_data`, `add_anomaly_data` and `create_validation_set`. Three first methods are similar in using. You need specify directory from you want adding data and optionality specify `file_format` (in default it is `mp4`). Here the example of adding data for model above:
```python
model.add_data("path/to/normal/data")
model.add_validation_data("path/to/normal/validation/data")
model.add_anomaly_data("path/to/abnormal/data")
```
This code will create 3 data sets for the model. It is mandatory to add training data to train the model, the rest if desired. Note that when the data is first initialized, there will be an AU extraction using mediapipe and this will take some time. However, after processing the video, the `pkl` files will be created in the specified directory containing the data already required for further work with these videos without reapplying to mediapipe.

Last method create validation set from train_data. It is necessary for validation model during training. You can specify `validation_size` (in default it is `0.2`).
```python
model.create_validation_set(validation_size=0.1)
```
### Set model name
This method allow you choose path and name with which the model will be saved (in default it is `default_model`).
```python
model.set_model_name("path/to/model/name")
```
### Training
After you adding all necessary data, you can start training the model. For this you need to call the `train` method. You can specify `save_model` (in default it is `False`) if you want save model in every epoch.
```python
model.train(save_model=True)
```
### Save best model
Trainer can memorize best model and save it. For this you need to call the `save_best_model` method.
```python
model.save_best_model()
```

## Test Model
In `test` module you can call `test` function.
```python
from test import test
test(source="path/to/your/video.mp4",
     model_path="path/to/your/model.pt",
     params_path="path/to/your/model/params.json",
     model_type="transformer",
     criterion="MSELoss")
```

In `source` you can also specify `0` and then the model will take data from your webcam.