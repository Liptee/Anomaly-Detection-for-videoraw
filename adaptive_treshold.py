import cv2
import json
import torch
import mediapipe as mp
from models import Transformer, RNN, LSTM
from utils import from_landmarks_to_array, load_data
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

mp_pose = mp.solutions.pose.Pose()


def start_experiment(path_to_data: str, models: list, parameters: list, type_models: list):
    if len(models) != len(parameters) or len(models) != len(type_models):
        raise ValueError("The number of models, parameters and type of models must be the same")

    for i in range(len(models)):
        model_path = models[i]
        params_path = parameters[i]
        model_type = type_models[i]

        with open(params_path) as f:
            params = json.load(f)

        model_classes = {
            "transformer": Transformer,
            "rnn": RNN,
            "lstm": LSTM
        }

        if model_type not in model_classes:
            raise ValueError("Model type must be one of: {}".format(model_classes.keys()))
        model = model_classes[model_type](params)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        normal_data = load_data(path_to_data + "/normal", "mp4")
        anomaly_data = load_data(path_to_data + "/anomaly", "mp4")
        losses = []
        y_true = []
        for video in tqdm(normal_data):
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                print("Error opening video stream or file")
                exit()

            sequence = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = mp_pose.process(frame)
                if results.pose_landmarks:
                    sequence.append(from_landmarks_to_array(results.pose_landmarks.landmark))
                else:
                    sequence = []

                if len(sequence) >= params["sequence_length"]:
                    y_true.append(0)
                    current_sequence = sequence[-params["sequence_length"]:]
                    current_sequence = torch.tensor(current_sequence, dtype=torch.float32)
                    current_sequence = current_sequence.view(current_sequence.shape[0],
                                                             current_sequence.shape[1] * current_sequence.shape[2])
                    current_sequence = current_sequence.unsqueeze(0)
                    outputs = model(current_sequence)
                    loss = torch.nn.MSELoss()(outputs, current_sequence).item()

                    losses.append(loss)

        for video in tqdm(anomaly_data):
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                print("Error opening video stream or file")
                exit()

            sequence = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = mp_pose.process(frame)
                if results.pose_landmarks:
                    sequence.append(from_landmarks_to_array(results.pose_landmarks.landmark))
                else:
                    sequence = []

                if len(sequence) >= params["sequence_length"]:
                    y_true.append(1)
                    current_sequence = sequence[-params["sequence_length"]:]
                    current_sequence = torch.tensor(current_sequence, dtype=torch.float32)
                    current_sequence = current_sequence.view(current_sequence.shape[0],
                                                             current_sequence.shape[1] * current_sequence.shape[2])
                    current_sequence = current_sequence.unsqueeze(0)
                    outputs = model(current_sequence)
                    loss = torch.nn.MSELoss()(outputs, current_sequence).item()

                    losses.append(loss)

        thresholds = np.arange(0.0, 0.001, 0.0000001)
        best_threshold = 0
        best_f1 = 0
        for threshold in tqdm(thresholds):
            y_pred = [1 if loss > threshold else 0 for loss in losses]
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print("Model: {}".format(model_path))
        print("Best threshold: {}".format(best_threshold))
        print("Best f1: {}".format(best_f1))

        params["threshold"] = best_threshold
        with open(params_path, "w") as f:
            json.dump(params, f)


if __name__ == "__main__":
    models = ["models/release/RGB/transformer/model_128_12_2_score_-0.0002.pt",
              "models/release/RGB/transformer/model_128_14_2_score_-0.0002.pt",
              "models/release/RGB/transformer/model_256_12_2_score_-1.6261.pt"]

    parameters = [i[:-3] + ".json" for i in models]
    type_models = [i.split("/")[3] for i in models]
    start_experiment("data/final_anomaly_set/RGB", models, parameters, type_models)
