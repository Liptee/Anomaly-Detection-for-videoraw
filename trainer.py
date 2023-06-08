from models import Transformer, CNN
from sklearn.model_selection import train_test_split
from torch import nn, optim
from utils import load_data, extract_sequential, make_samples
import pickle
import numpy as np


class Trainer:
    def __init__(self,
                 params: dict,
                 model: str = "transformer",
                 criterion=nn.MSELoss(),
                 learning_rate=0.001,
                 optimizer=optim.Adam,
                 device='cpu',
                 batch_size=32,
                 num_epochs=10,
                 sequence_length=12):

        if model == "transformer":
            self.model = Transformer(params)
        elif model == "cnn":
            self.model = CNN(params)
        else:
            raise ValueError("model should be either transformer or cnn")
        self.params = params
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.model.to(self.device)
        self.sequence_length = sequence_length

        self.data = None
        self.val_data = None
        self.anomaly_data = None

    def add_data(self, path_to_dir, file_format: str = "mp4"):
        """
        Adds data to the model
        :param path_to_dir: path to directory containing data
        :param file_format: file format of data
        :return: None
        """
        if self.data is None:
            self.data = []
        files_list = load_data(path_to_dir, file_format)
        metadata_list = load_data(path_to_dir, "pkl")

        for path in files_list:
            name = path.split(".")[0]
            if f"{name}.pkl" in metadata_list:
                print(f"Extracting data from {name}.pkl...")
                with open(f"{name}.pkl", "rb") as f:
                    sequential = pickle.load(f)
            else:
                sequential = extract_sequential(path, make_mirrors=True)
                with open(f"{name}.pkl", "wb") as f:
                    pickle.dump(sequential, f)

            data = make_samples(sequential, self.sequence_length)
            data = np.array(data)
            self.data.extend(data)
            print(f"Size of train data: {np.array(self.data).shape}")

    def add_validation_data(self, path_to_dir, file_format: str = "mp4"):
        """
        Adds validation data to the model
        :param path_to_dir: path to directory containing data
        :param file_format: file format of data
        :return: None
        """
        if self.val_data is None:
            self.val_data = []
        files_list = load_data(path_to_dir, file_format)
        metadata_list = load_data(path_to_dir, "pkl")

        for path in files_list:
            name = path.split(".")[0]
            if f"{name}.pkl" in metadata_list:
                print(f"Extracting data from {name}.pkl...")
                with open(f"{name}.pkl", "rb") as f:
                    sequential = pickle.load(f)
            else:
                sequential = extract_sequential(path, make_mirrors=True)
                with open(f"{name}.pkl", "wb") as f:
                    pickle.dump(sequential, f)

            data = make_samples(sequential, self.sequence_length)
            data = np.array(data)
            self.val_data.extend(data)
            print(f"Size of validation data: {np.array(self.val_data).shape}")

    def add_anomaly_data(self, path_to_dir, file_format: str = "mp4"):
        """
        Adds anomaly data to the model
        :param path_to_dir: path to directory containing data
        :param file_format: file format of data
        :return: None
        """
        if self.anomaly_data is None:
            self.anomaly_data = []
        files_list = load_data(path_to_dir, file_format)
        metadata_list = load_data(path_to_dir, "pkl")

        for path in files_list:
            name = path.split(".")[0]
            if f"{name}.pkl" in metadata_list:
                print(f"Extracting data from {name}.pkl...")
                with open(f"{name}.pkl", "rb") as f:
                    sequential = pickle.load(f)
            else:
                sequential = extract_sequential(path, make_mirrors=True)
                with open(f"{name}.pkl", "wb") as f:
                    pickle.dump(sequential, f)

            data = make_samples(sequential, self.sequence_length)
            data = np.array(data)
            self.anomaly_data.extend(data)
            print(f"Size of anomaly data: {np.array(self.anomaly_data).shape}")

    def create_validatation_set(self, test_size=0.2):
        """
        Splits the data into train and validation sets
        :param test_size: size of validation set
        :return: None
        """
        self.data, val_data = train_test_split(self.data, test_size=test_size)
        if self.val_data is None:
            self.val_data = val_data
        else:
            self.val_data.extend(val_data)
        print(f"Size of train data: {np.array(self.data).shape}")
        print(f"Size of validation data: {np.array(self.val_data).shape}")


if __name__ == "__main__":
    params = {
        "input_size": 32,
        "num_heads": 4,
        "hidden_size": 128,
        "dropout": 0.2,
        "num_layers": 3
    }
    trainer = Trainer(params)
    trainer.add_data("test_data")
    trainer.add_data("test_data_2")
    trainer.make_validatation_set()
