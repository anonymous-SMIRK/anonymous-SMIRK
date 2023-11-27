import os

def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def create_LRS3_lists(lrs3_path, lrs3_landmarks_path):
    from sklearn.model_selection import train_test_split
    import pickle
    trainval_folder_list = list(os.listdir(f"{lrs3_path}/trainval"))
    train_folder_list, val_folder_list = train_test_split(trainval_folder_list, test_size=0.2, random_state=42)
    test_folder_list = list(os.listdir(f"{lrs3_path}/test"))



    def gather_LRS3_split(folder_list, split="trainval"):
        list_ = []
        for folder in folder_list:
            for file in os.listdir(os.path.join(f"{lrs3_path}/{split}", folder)):
                if file.endswith(".txt"):
                    file_without_extension = file.split(".")[0]
                    file_inner_path = f"{split}/{folder}/{file_without_extension}"

                    landmarks_filename = os.path.join(lrs3_landmarks_path, file_inner_path+".pkl")

                    valid = True
                    with open(landmarks_filename, "rb") as pkl_file:
                        landmarks = pickle.load(pkl_file)
                        preprocessed_landmarks = landmarks_interpolate(landmarks)
                        if preprocessed_landmarks is None:
                            valid = False

                    mediapipe_landmarks_filepath = os.path.join(lrs3_path, file_inner_path+".npy")
                    if not os.path.exists(mediapipe_landmarks_filepath):
                        valid = False
                    if os.path.exists(landmarks_filename) and valid:
                        subject = folder
                        list_.append([os.path.join(lrs3_path, file_inner_path + ".mp4"), os.path.join(lrs3_landmarks_path, file_inner_path+".pkl"), 
                                      mediapipe_landmarks_filepath,
                                      subject])
        return list_

    train_list = gather_LRS3_split(train_folder_list, split="trainval")
    val_list = gather_LRS3_split(val_folder_list, split="trainval")
    test_list = gather_LRS3_split(test_folder_list, split="test")

    print(len(train_list), len(val_list), len(test_list))

    pickle.dump([train_list,val_list,test_list], open(f"assets/LRS3_lists.pkl", "wb"))
