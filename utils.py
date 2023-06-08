import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from glob import glob


mp_pose = mp.solutions.pose.Pose()


def load_data(path, file_type):
    """
    Loads data from a directory into a list of tensors
    :param path: path to directory
    :param file_type: file type to load
    :return: list of tensors
    """
    files = glob(os.path.join(path, f"*.{file_type}"))
    return files


def extract_sequential(path_to_video, make_mirrors=False):
    """
    Extracts sequential data from a video
    :param path_to_video: path to video
    :return
    """
    results = []
    one_sequnce = []
    mirror_sequnce = []
    cap = cv2.VideoCapture(path_to_video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(length), desc=f"Mediapipe is extracting data from {path_to_video}..."):
        _, frame = cap.read()
        result = mp_pose.process(frame)

        if result.pose_landmarks:
            arr = from_landmarks_to_array(result.pose_landmarks.landmark)
            one_sequnce.append(arr)
        else:
            if len(one_sequnce) > 1:
                results.append(one_sequnce)
            one_sequnce = []

        if make_mirrors:
            mirror = frame[:,::-1]
            result = mp_pose.process(mirror)

            if result.pose_landmarks:
                arr = from_landmarks_to_array(result.pose_landmarks.landmark)
                mirror_sequnce.append(arr)
            else:
                if len(mirror_sequnce) > 1:
                    results.append(mirror_sequnce)
                mirror_sequnce = []

    if len(one_sequnce) > 1:
        results.append(one_sequnce)
    if len(mirror_sequnce) > 1 and make_mirrors:
        results.append(mirror_sequnce)

    return results


def from_landmarks_to_array(landmarks):
    """
    Converts landmarks to a numpy array
    :param landmarks: landmarks
    :return: numpy array
    """
    res = np.zeros((32, 4))

    # Here we use the nose landmark as the coefficient for the whole pose
    # We need to exclude the influence of the scene from the data,
    # so we take the coordinates of the nose
    # and subtract them from the coordinates of all other landmarks

    nose_landmark = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
    landmarks = landmarks[1:]
    for i, landmark in enumerate(landmarks):
        res[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        res[i][0] -= nose_landmark[0]
        res[i][1] -= nose_landmark[1]
        res[i][2] -= nose_landmark[2]

    return res


def make_samples(sequences, sequence_length):
    """
    Makes samples from sequences
    :param sequences: list of sequences
    :param sequence_length: length of each sequence
    :return: list of samples
    """
    samples = []
    for sequence in sequences:
        for i in range(len(sequence) - sequence_length):
            samples.append(sequence[i:i + sequence_length])
    return samples
