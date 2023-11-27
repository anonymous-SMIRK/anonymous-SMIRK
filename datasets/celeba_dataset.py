import os
import numpy as np
import cv2
import sys
from datasets.base_dataset import BaseDataset
import sys

class CelebADataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.keys = list(data_list.keys())
        self.name = 'CelebA'

    def __len__(self):
        return len(self.keys)

    def __getitem_aux__(self, index):
        key = self.keys[index]

        fan_landmarks_path = os.path.join(self.config.dataset.CelebA_fan_landmarks_path)
        mediapipe_landmarks_path = os.path.join(self.config.dataset.CelebA_mediapipe_landmarks_path)
        folder_path = os.path.join(self.config.dataset.CelebA_path)

        files_list = self.data_list[key]

        fan_landmarks_list = []
        mediapipe_landmarks_list = []
        for file in files_list:
            fan_landmarks_files = os.path.join(fan_landmarks_path, file.replace('.jpg','.npy').replace(".png",".npy"))
            fan_landmarks_list.append(fan_landmarks_files)

            mediapipe_landmarks_files = os.path.join(mediapipe_landmarks_path, file.replace('.jpg','.npy').replace(".png",".npy"))
            mediapipe_landmarks_list.append(mediapipe_landmarks_files)

        files_list = [os.path.join(folder_path, file) for file in files_list]

        data_dict = self.sample_frames(files_list, fan_landmarks_list, mediapipe_landmarks_list)

        data_dict['subject'] = ""
        data_dict['filename'] = ""

        return data_dict



def get_datasets_CelebA(config=None):
    file = "datasets/identity_CelebA.txt"
    with open(file) as f:
        lines = f.readlines()

    subjects = [x.split()[1].strip() for x in lines]
    files = [x.split()[0] for x in lines]

    train_dict = {}
    num_files = 0
    for subject, file in zip(subjects, files):
        if subject not in train_dict:
            train_dict[subject] = []

        # if not os.path.exists(os.path.join(config.dataset.CelebA_mediapipe_landmarks_path, file.replace('.jpg','.npy').replace(".png",".npy"))):
        #     continue

        train_dict[subject].append(file)
        num_files += 1

    print("Number of subjects CeleBA: ", len(train_dict.keys()))
    print("Number of files CeleBA: ", num_files)

    return CelebADataset(train_dict, config)



if __name__ == '__main__':

    from omegaconf import OmegaConf
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    # Remove the configuration file name from sys.argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # merge config with cli args
    conf.merge_with_cli()

    dataset, train_dict = get_datasets_CelebA(conf)
    import matplotlib.pyplot as plt

    for key in train_dict.keys():

        grid = []
        for im in train_dict[key]:
            image = os.path.join(conf.dataset.CelebA_path, im.replace('.npy','.jpg'))
            image = cv2.imread(image)
            grid.append(image)


        # Number of images in the grid
        num_images = len(grid)

        rows = 5
        cols = 5

        # Creating subplots

        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))  # Adjust the size as needed

        # Flatten the axes array for easy iteration if it's 2D
        axes = axes.flatten()

        # Iterate over the grid and axes simultaneously
        for ax, img in zip(axes, grid):
            # Convert the image from BGR (OpenCV default) to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.axis('off')  # Turn off axis

        # If there are more subplots than images, turn off the extra subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

