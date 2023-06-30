from models import Transformer, RNN, LSTM
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from utils import load_data, extract_sequential, make_samples, return_statisctic_for_list
import pickle
import numpy as np
import json


class Trainer:
    def __init__(self,
                 params: dict,
                 model="transformer",
                 criterion=nn.MSELoss(),
                 learning_rate=0.001,
                 optimizer=optim.Adam,
                 device='cpu',
                 batch_size=32,
                 num_epochs=10,
                 sequence_length=12):

        model_classes = {
            "transformer": Transformer,
            "rnn": RNN,
            "lstm": LSTM
        }
        if model in model_classes:
            self.model = model_classes[model](params)
        else:
            raise ValueError(f"Model {model} not found")

        params["sequence_length"] = sequence_length
        self.model_type = model
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
        self.output_model_name = "model"
        self.best_model = None
        self.best_params = None
        self.best_loss = 1000000.0
        self.best_score = -100000

    def add_data(self, path_to_dir, file_format: str = "mp4", rewrite=False):
        if self.data is None:
            self.data = []
        files_list = load_data(path_to_dir, file_format)
        metadata_list = load_data(path_to_dir, "pkl")

        for path in files_list:
            name = path.split(".")[-2]
            if f"{name}.pkl" in metadata_list and not rewrite:
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

    def add_validation_data(self, path_to_dir, file_format: str = "mp4", rewrite=False):
        if self.val_data is None:
            self.val_data = []
        files_list = load_data(path_to_dir, file_format)
        metadata_list = load_data(path_to_dir, "pkl")

        for path in files_list:
            name = path.split(".")[-2]
            if f"{name}.pkl" in metadata_list and not rewrite:
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

    def add_anomaly_data(self, path_to_dir, file_format: str = "mp4", rewrite=False):
        if self.anomaly_data is None:
            self.anomaly_data = []
        files_list = load_data(path_to_dir, file_format)
        metadata_list = load_data(path_to_dir, "pkl")

        for path in files_list:
            name = path.split(".")[-2]
            if f"{name}.pkl" in metadata_list and not rewrite:
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

    def create_validation_set(self, test_size=0.2):
        self.data, val_data = train_test_split(self.data, test_size=test_size)
        if self.val_data is None:
            self.val_data = val_data
        else:
            print("Warning: validation data already exists. Appending new data...")
            self.val_data.extend(val_data)
        print(f"Size of train data: {np.array(self.data).shape}")
        print(f"Size of validation data: {np.array(self.val_data).shape}")

    def set_output_model_name(self, name):
        self.output_model_name = name

    def save_best_model(self):
        if self.best_model is None:
            raise ValueError("No best model to save")

        if self.anomaly_data:
            torch.save(self.best_model.state_dict(), f"{self.output_model_name}_score_{self.best_score:.4f}.pt")
            with open(f"{self.output_model_name}_score_{self.best_score:.4f}.json", "w") as f:
                json.dump(self.best_params, f)
        else:
            torch.save(self.best_model.state_dict(), f"{self.output_model_name}_loss_{self.best_loss:.4f}.pt")
            with open(f"{self.output_model_name}_loss_{self.best_loss:.4f}.json", "w") as f:
                json.dump(self.best_params, f)

    def train(self, save_model: bool = False):
        if self.data is None:
            raise ValueError("No data added to the model")

        data = np.array(self.data, dtype=np.float32)
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))

        dataset = MyDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.val_data:
            val_data = np.array(self.val_data, dtype=np.float32)
            val_data = val_data.reshape((val_data.shape[0], val_data.shape[1], val_data.shape[2] * val_data.shape[3]))
            val_dataset = MyDataset(np.array(val_data))
            val_dataloader = DataLoader(val_dataset, batch_size=1)

        if self.anomaly_data:
            anomaly_data = np.array(self.anomaly_data, dtype=np.float32)
            anomaly_data = anomaly_data.reshape(
                (anomaly_data.shape[0], anomaly_data.shape[1], anomaly_data.shape[2] * anomaly_data.shape[3]))
            anomaly_dataset = MyDataset(np.array(anomaly_data))
            anomaly_dataloader = DataLoader(anomaly_dataset, batch_size=1)

        for epoch in range(self.num_epochs):
            train_losses = []
            val_losses = []
            anomaly_losses = []
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.model.train()
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            train_mean, train_median, train_std, train_min, train_max = return_statisctic_for_list(train_losses)
            print(f"Train loss: {train_mean:.4f} +- {train_std:.4f}")

            if self.val_data:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        batch = batch.to(self.device)
                        output = self.model(batch)
                        loss = self.criterion(output, batch)
                        val_losses.append(loss.item())
                val_mean, val_median, val_std, val_min, val_max = return_statisctic_for_list(val_losses)
                print(f"Validation loss: {val_mean:.4f} +- {val_std:.4f}")

            if self.anomaly_data:
                self.model.eval()
                with torch.no_grad():
                    for batch in anomaly_dataloader:
                        batch = batch.to(self.device)
                        output = self.model(batch)
                        loss = self.criterion(output, batch)
                        anomaly_losses.append(loss.item())
                anomaly_mean, anomaly_median, anomaly_std, anomaly_min, anomaly_max = return_statisctic_for_list(
                    anomaly_losses)
                print(f"Anomaly loss: {anomaly_mean:.4f} +- {anomaly_std:.4f}")

            if self.val_data:
                self.params["mean"] = val_mean
                self.params["median"] = val_median
                self.params["std"] = val_std
                self.params["min"] = val_min
                self.params["max"] = val_max
            else:
                self.params["mean"] = train_mean
                self.params["median"] = train_median
                self.params["std"] = train_std
                self.params["min"] = train_min
                self.params["max"] = train_max

            if not self.anomaly_data:
                score = self.params["mean"]
                if score < self.best_loss:
                    self.best_loss = score
                    self.best_params = self.params
                    self.best_model = self.model
                if save_model:
                    torch.save(self.model.state_dict(), f"{self.output_model_name}_loss_{score:.4f}.pt")
                    with open(f"{self.output_model_name}_loss_{score:.4f}.json", "w") as f:
                        json.dump(self.params, f)

            else:
                sorted_anomaly = sorted(anomaly_losses)
                sorted_normal = sorted(val_losses)
                score = np.mean(sorted_anomaly[:200]) - np.mean(sorted_normal[-200:])
                print(f"Score: {score}")
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = self.params
                    self.best_model = self.model

                if save_model:
                    torch.save(self.model.state_dict(), f"{self.output_model_name}_score_{score:.4f}.pt")
                    with open(f"{self.output_model_name}_score_{score}.json", "w") as f:
                        json.dump(self.params, f)
            print("-" * 50)


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)