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
    :return list of sequences. Sequence is a list of frames. Frames are numpy arrays with shape (32, 4)
    """
    results = []
    sequence = []
    m_sequence = []
    cap = cv2.VideoCapture(path_to_video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(length), desc=f"Mediapipe is extracting data from {path_to_video}..."):
        _, frame = cap.read()

        result = mp_pose.process(frame)
        if result.pose_landmarks:
            sequence.append(from_landmarks_to_array(result.pose_landmarks.landmark))
        else:
            if len(sequence) > 1:
                results.append(sequence)
            sequence = []

        if make_mirrors:
            m_frame = frame[:, ::-1]
            m_result = mp_pose.process(m_frame)
            if m_result.pose_landmarks:
                m_sequence.append(from_landmarks_to_array(m_result.pose_landmarks.landmark))
            else:
                if len(m_sequence) > 1:
                    results.append(m_sequence)
                m_sequence = []

    if len(sequence) > 1:
        results.append(sequence)
    if len(m_sequence) > 1:
        results.append(m_sequence)
    return results


def from_landmarks_to_array(landmarks):
    """
    Converts landmarks to a numpy array
    :param landmarks: landmarks
    :return: numpy array with shape (32, 4)
    """
    res = np.zeros((32, 4))

    # Here we use the nose landmark as the coefficient for the whole pose
    # We need to exclude the influence of the scene from the data,
    # so we take the coordinates of the nose
    # and subtract them from the coordinates of all other landmarks

    nose_landmark = [landmarks[0].x-0.5, landmarks[0].y-0.2, landmarks[0].z+0.5]
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
    :return: list of samples. Every sample is a sequence of length sequence_length
    """
    samples = []
    for sequence in sequences:
        for i in range(len(sequence) - sequence_length + 1):
            samples.append(sequence[i:i + sequence_length])
    return samples


def return_statisctic_for_list(lst):
    mean = np.mean(lst)
    median = np.median(lst)
    std = np.std(lst)
    minimum = np.min(lst)
    maximum = np.max(lst)
    return mean, median, std, minimum, maximum
