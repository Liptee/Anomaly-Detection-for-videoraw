import cv2
import json
import torch
import mediapipe as mp
from models import Transformer, CNN
from utils import from_landmarks_to_array


mp_pose = mp.solutions.pose.Pose()

def test(source, model_path: str, params_path, model_type="transformer", criterion=torch.nn.MSELoss()):
    with open(params_path) as f:
        params = json.load(f)
    if model_type == "transformer":
        model = Transformer(params)
    elif model_type == "cnn":
        model = CNN(params)
    model.load_state_dict(torch.load(model_path))

    print(params)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    threshold = params["max"]

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

        state = "None"
        if len(sequence) >= params["sequence_length"]:
            state = "Normal"
            current_sequence = sequence[-params["sequence_length"]:]
            current_sequence = torch.tensor(current_sequence, dtype=torch.float32)
            if model_type == "transformer":
                current_sequence = current_sequence.view(current_sequence.shape[0], current_sequence.shape[1] * current_sequence.shape[2])
            elif model_type == "cnn":
                current_sequence = current_sequence.view(current_sequence.shape[2], current_sequence.shape[0], current_sequence.shape[1])
            outputs = model(current_sequence)
            loss = criterion(outputs, current_sequence)
            if loss.item() > threshold:
                state = "Anomaly"

        frame = cv2.putText(frame, state, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

test(0, "models/test_cnn_loss_0.0950.pt", "models/test_cnn_loss_0.0950.json", model_type="cnn")
