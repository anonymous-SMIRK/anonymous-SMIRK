import os
import pickle
from .data_utils import landmarks_interpolate
from datasets.base_dataset import BaseDataset
import numpy as np

class MEADDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'MEAD'

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        sample = self.data_list[index]

        landmarks_filename = sample[1]
        video_path = sample[0]
        mediapipe_landmarks_path = sample[2]

        if not os.path.exists(landmarks_filename):
            raise Exception('Video %s has no landmarks'%(sample))

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                raise Exception('Video %s has no landmarks'%(sample))

        mediapipe_landmarks = np.load(mediapipe_landmarks_path)

        data_dict = self.sample_frames(video_path, preprocessed_landmarks, mediapipe_landmarks)

        data_dict['subject'] = ""
        data_dict['filename'] = ""


        return data_dict
    


def get_datasets_MEAD(config=None):

    # Assuming you're currently in the directory where the files are located
    files = [f for f in os.listdir(config.dataset.MEAD_landmarks_path)]

    # subjects = list(set([f.split('_')[0] for f in files]))
    # print(subjects)
    # Split the subjects into train, validation, and test sets
    # train_subjects, test_subjects = train_test_split(subjects, test_size=0.1, random_state=42)
    # train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.1, random_state=42) 
    # print("Train Subjects:", sorted(train_subjects))
    # print("Validation Subjects:", sorted(val_subjects))
    # print("Test Subjects:", sorted(test_subjects))

    # this is the split used in the paper, randomly selected
    train_subjects = ['M003', 'M007', 'M009', 'M011', 'M012', 'M019', 'M024', 'M025', 'M026', 'M027', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M037', 'M039', 'M040', 'M041', 'W009', 'W011', 'W014', 'W015', 'W016', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W035', 'W036', 'W037', 'W038', 'W040']
    val_subjects = ['M013', 'M023', 'M042', 'W018', 'W028']
    test_subjects = ['M005', 'M022', 'M028', 'W029', 'W033']

    # assert each subject is in exactly one split
    assert len(set(train_subjects).intersection(val_subjects)) == 0
    assert len(set(train_subjects).intersection(test_subjects)) == 0
    assert len(set(val_subjects).intersection(test_subjects)) == 0

    train_list = []
    for file in files:
        if file.split('_')[0] in train_subjects:
            landmarks_path = os.path.join(config.dataset.MEAD_landmarks_path, file.split(".")[0] + ".pkl")
            folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
            mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_path, file.split(".")[0] + ".npy")
            train_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    val_list = []
    for file in files:
        if file.split('_')[0] in val_subjects:
            landmarks_path = os.path.join(config.dataset.MEAD_landmarks_path, file.split(".")[0] + ".pkl")
            folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
            mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_path, file.split(".")[0] + ".npy")
            val_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    test_list = []
    for file in files:
        if file.split('_')[0] in test_subjects:
            landmarks_path = os.path.join(config.dataset.MEAD_landmarks_path, file.split(".")[0] + ".pkl")
            folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
            mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_path, file.split(".")[0] + ".npy")
            test_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    # print("Train Files:", len(train_list))
    # print("Validation Files:", len(val_list))
    # print("Test Files:", len(test_list))

    return MEADDataset(train_list, config), MEADDataset(val_list, config, test=True), MEADDataset(test_list, config, test=True)





