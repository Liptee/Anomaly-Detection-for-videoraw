import cv2
import json
import torch
import mediapipe as mp
from models import Transformer, RNN, LSTM
from utils import from_landmarks_to_array, load_data
from sklearn.metrics import classification_report
from tqdm import tqdm

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
        predictions = []
        y_true = []
        for video in tqdm(normal_data):
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                print("Error opening video stream or file")
                exit()

            threshold = params["threshold"]
            sequence = []
            losses = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = mp_pose.process(frame)
                if results.pose_landmarks:
                    sequence.append(from_landmarks_to_array(results.pose_landmarks.landmark))
                else:
                    sequence = []

                state = "None"
                if len(sequence) >= params["sequence_length"]:
                    y_true.append(0)
                    state = "Normal"
                    current_sequence = sequence[-params["sequence_length"]:]
                    current_sequence = torch.tensor(current_sequence, dtype=torch.float32)
                    current_sequence = current_sequence.view(current_sequence.shape[0],
                                                             current_sequence.shape[1] * current_sequence.shape[2])
                    current_sequence = current_sequence.unsqueeze(0)
                    outputs = model(current_sequence)
                    loss = torch.nn.MSELoss()(outputs, current_sequence).item()

                    losses.append(loss)

                    if loss > threshold:
                        predictions.append(1)
                        state = "Anomaly"
                    else:
                        predictions.append(0)


                frame = cv2.putText(frame, state, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        for video in tqdm(anomaly_data):
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                print("Error opening video stream or file")
                exit()

            threshold = params["threshold"]
            sequence = []
            losses = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = mp_pose.process(frame)
                if results.pose_landmarks:
                    sequence.append(from_landmarks_to_array(results.pose_landmarks.landmark))
                else:
                    sequence = []

                state = "None"
                if len(sequence) >= params["sequence_length"]:
                    y_true.append(1)
                    state = "Normal"
                    current_sequence = sequence[-params["sequence_length"]:]
                    current_sequence = torch.tensor(current_sequence, dtype=torch.float32)
                    current_sequence = current_sequence.view(current_sequence.shape[0],
                                                             current_sequence.shape[1] * current_sequence.shape[2])
                    current_sequence = current_sequence.unsqueeze(0)
                    outputs = model(current_sequence)
                    loss = torch.nn.MSELoss()(outputs, current_sequence).item()

                    losses.append(loss)

                    if loss > threshold:
                        predictions.append(1)
                        state = "Anomaly"
                    else:
                        predictions.append(0)


                frame = cv2.putText(frame, state, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print("Classification report for model: {}".format(model_path))
        print(classification_report(y_true, predictions))
