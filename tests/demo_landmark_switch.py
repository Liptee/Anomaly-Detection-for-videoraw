import cv2
import mediapipe as mp
from utils import from_landmarks_to_array

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose.Pose()

def test(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = mp_pose.process(frame)
        if results.pose_landmarks:
            arr = from_landmarks_to_array(results.pose_landmarks.landmark)
            results.pose_landmarks.landmark[0].x = 0.5
            results.pose_landmarks.landmark[0].x = 0.2
            results.pose_landmarks.landmark[0].x = -0.5
            for i in range(1, 33):
                results.pose_landmarks.landmark[i].x = arr[i - 1][0]
                results.pose_landmarks.landmark[i].y = arr[i - 1][1]
                results.pose_landmarks.landmark[i].z = arr[i - 1][2]
                results.pose_landmarks.landmark[i].visibility = 1
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

test("cptn.mp4")
