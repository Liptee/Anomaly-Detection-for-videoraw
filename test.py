import cv2
import json
import torch
import mediapipe as mp
from models import Transformer, CNN, FC_CNN, RNN
from utils import from_landmarks_to_array
import matplotlib.pyplot as plt


mp_pose = mp.solutions.pose.Pose()

def test(source,
         model_path: str,
         params_path,
         model_type="transformer",
         criterion=torch.nn.MSELoss(),
         transpose=(2, 0, 1)):
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
    threshold = 0.00015
    minimum = params["min"]

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
            state = "Normal"
            current_sequence = sequence[-params["sequence_length"]:]
            current_sequence = torch.tensor(current_sequence, dtype=torch.float32)
            if model_type == "transformer" or model_type == "rnn":
                current_sequence = current_sequence.view(current_sequence.shape[0], current_sequence.shape[1] * current_sequence.shape[2])
            elif model_type == "cnn" or model_type == "fc_cnn":
                current_sequence = current_sequence.view(current_sequence.shape[transpose[0]], current_sequence.shape[transpose[1]], current_sequence.shape[transpose[2]])
            # add dimension in current_sequence
            current_sequence = current_sequence.unsqueeze(0)
            outputs = model(current_sequence)
            loss = criterion(outputs, current_sequence).item()

            losses.append(loss)

            if loss > threshold:
                state = "Anomaly"


        frame = cv2.putText(frame, state, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #create plot from losses and save as png file. Also draw in this plot threshold line, labels, x and y axis and other stuff.
    plt.plot(losses)
    plt.axhline(y=mean, color='red', linestyle='-')
    plt.axhline(y=threshold, color='orange', linestyle='-')
    plt.axhline(y=minimum, color='green', linestyle='-')
    plt.xlabel("Frame")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.savefig(f"{name}.png".format(name))

    # clear plot
    plt.clf()


if __name__ == "__main__":
    from utils import load_data
    anomalys = []
    normals = []
    for i in range(1, 4):
        anomalys.extend(load_data("data/final_test_anomaly/obj_{}/IR/anomaly".format(i), "mp4"))
        normals.extend(load_data("data/final_test_anomaly/obj_{}/IR/normal".format(i), "mp4"))
    for video in normals:
        print(video)
        test(video,
             "models/model_64_score_-0.0348.pt",
             "models/model_64_score_-0.0348.json",
             model_type="rnn")
