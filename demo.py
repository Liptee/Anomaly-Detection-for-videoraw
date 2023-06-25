import cv2
import json
import torch
import mediapipe as mp
from models import Transformer, CNN, FC_CNN, RNN
from utils import from_landmarks_to_array
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose.Pose()

def demo(source,
         model_path: str,
         params_path: str,
         model_type="transformer",
         criterion=torch.nn.MSELoss(),
         transpose=(2, 0, 1),
         scale=2.0):
    if source == 0:
        name = "webcam"
    else:
        name = source.split(".")[0]

    with open(params_path) as f:
        params = json.load(f)

    model_classes = {
        "transformer": Transformer,
        "cnn": CNN,
        "fc_cnn": FC_CNN,
        "rnn": RNN
    }

    if model_type not in model_classes:
        raise ValueError("Model type must be one of: {}".format(model_classes.keys()))
    model = model_classes[model_type](params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    mean = params["mean"]
    threshold = 0.00013
    maximun = params["max"]

    ret, frame = cap.read()
    if not ret:
        print("Error opening video stream or file")
        exit()

    height = frame.shape[0]
    width = frame.shape[1]
    HEIGHT = int(height // scale)
    WIDTH = int(width // scale)
    fheight = HEIGHT * 2
    fwidth = WIDTH * 2

    sequence = []
    losses = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = mp_pose.process(frame)

        final_frame = np.zeros((fheight, fwidth, 3), np.uint8)
        original_landmarks_frame = np.zeros((height, width, 3), np.uint8)
        predicted_landmarks_frame = np.zeros((height, width, 3), np.uint8)

        if results.pose_landmarks:
            arr = from_landmarks_to_array(results.pose_landmarks.landmark)
            results.pose_landmarks.landmark[0].x = 0.5
            results.pose_landmarks.landmark[0].x = 0.2
            results.pose_landmarks.landmark[0].x = -0.5
            for i in range(9, 33):
                results.pose_landmarks.landmark[i].x = arr[i - 9][0]
                results.pose_landmarks.landmark[i].y = arr[i - 9][1]
                results.pose_landmarks.landmark[i].z = arr[i - 9][2]
                results.pose_landmarks.landmark[i].visibility = 1
            mp_drawing.draw_landmarks(
                original_landmarks_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            sequence.append(arr)

        else:
            sequence = []

        if len(sequence) >= params["sequence_length"]:
            state = "Normal"
            current_sequence = sequence[-params["sequence_length"]:]
            current_sequence = np.array(current_sequence)
            current_sequence = torch.tensor(current_sequence, dtype=torch.float32)
            if model_type == "transformer" or model_type == "rnn":
                current_sequence = current_sequence.view(current_sequence.shape[0],
                                                         current_sequence.shape[1] * current_sequence.shape[2])
            elif model_type == "cnn" or model_type == "fc_cnn":
                current_sequence = current_sequence.view(current_sequence.shape[transpose[0]],
                                                         current_sequence.shape[transpose[1]],
                                                         current_sequence.shape[transpose[2]])

            # add dimension in current_sequence
            current_sequence = current_sequence.unsqueeze(0)
            outputs = model(current_sequence)
            loss = criterion(outputs, current_sequence).item()
            losses.append(loss)

            if model_type == "transformer" or model_type == "rnn":
                last_frame = outputs[0][-1].detach().numpy()
                last_frame = last_frame.reshape(24, 3)
            if model_type == "cnn" or model_type == "fc_cnn":
                last_frame = outputs[0].detach().numpy()
                last_frame = last_frame.transpose(1, 2, 0)
                last_frame = last_frame[-1]

            for i in range(33):
                # print(f"x{i} before: {results.pose_landmarks.landmark[i].x}")
                # print(f"y{i} before: {results.pose_landmarks.landmark[i].y}")
                # print(f"z{i} before: {results.pose_landmarks.landmark[i].z}")
                if i > 9:
                    results.pose_landmarks.landmark[i].x = last_frame[i-9][0]
                    results.pose_landmarks.landmark[i].y = last_frame[i-9][1]
                    results.pose_landmarks.landmark[i].z = last_frame[i-9][2]
                # print(f"x{i} after: {results.pose_landmarks.landmark[i].x}")
                # print(f"y{i} after: {results.pose_landmarks.landmark[i].y}")
                # print(f"z{i} after: {results.pose_landmarks.landmark[i].z}")
                # print("-"*50)
            mp_drawing.draw_landmarks(
                predicted_landmarks_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        original_landmarks_frame = cv2.resize(original_landmarks_frame, (WIDTH, HEIGHT))
        predicted_landmarks_frame = cv2.resize(predicted_landmarks_frame, (WIDTH, HEIGHT))

        # draw losses plot for plot_frame with matplotlib
        fig, ax = plt.subplots()
        plt.plot(losses)
        plt.axhline(y=threshold, color='orange', linestyle='-')
        plt.axhline(y=mean, color='red', linestyle='-')
        plt.xlabel("Frame")
        plt.ylabel("Loss")
        plt.title("Losses")
        canvas = FigureCanvas(fig)
        canvas.draw()
        graph_image = np.array(canvas.renderer.buffer_rgba())
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)
        graph_image = cv2.resize(graph_image, (WIDTH, HEIGHT))
        plt.close(fig)

        final_frame[:HEIGHT, :WIDTH] = frame
        final_frame[:HEIGHT, WIDTH:] = graph_image
        final_frame[HEIGHT:, :WIDTH] = original_landmarks_frame
        final_frame[HEIGHT:, WIDTH:] = predicted_landmarks_frame

        cv2.imshow("Demo", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # close cap
    cap.release()


demo(0,
     "models/model_256_score_-0.0200.pt",
     "models/model_256_score_-0.0200.json",
     model_type="rnn",
     criterion=torch.nn.MSELoss(),
     transpose=(2, 0, 1),
     scale=1.0)
