import os
import torch
from datasets.base_dataset import BaseDataset


class FFHQDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'FFHQ'

    def __getitem_aux__(self, index):
        data_dict = self.sample_frames([self.data_list[index][0]], [self.data_list[index][1]], [self.data_list[index][2]])

        data_dict['subject'] = ""
        data_dict['filename'] = ""
        
        return data_dict


    
def get_datasets_FFHQ(config):
    if config.K > 1:
        print('Warning: K > 1 not supported for FFHQ dataset. Make sure you have FFHQ percentage set to 0.')

    train_list = []

    for image in os.listdir(config.dataset.FFHQ_path):
        if image.endswith(".png"):
            image_path = os.path.join(config.dataset.FFHQ_path, image)
            fan_landmarks_path = os.path.join(config.dataset.FFHQ_fan_landmarks_path, image.split(".")[0] + ".npy")
            mediapipe_landmarks_path = os.path.join(config.dataset.FFHQ_mediapipe_landmarks_path, image.split(".")[0] + ".npy")

            train_list.append([image_path, fan_landmarks_path, mediapipe_landmarks_path])

    dataset = FFHQDataset(train_list, config, test=False)
    return dataset






if __name__ == '__main__':

    from omegaconf import OmegaConf
    import sys
    from tqdm import tqdm

    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    # Remove the configuration file name from sys.argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # merge config with cli args
    conf.merge_with_cli()

    train_dataset = get_datasets_FFHQ(conf)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=8)

    for i in tqdm(range(len(loader))):
        sample = train_dataset[i]

        # image = sample['img'].squeeze().numpy().transpose(1,2,0).copy()
        # masked_image = sample['masked_img'].squeeze().numpy().transpose(1,2,0)
        # landmarks = sample['landmarks'].squeeze().numpy()

        # landmarks = landmarks * 112 + 112
        # # # plot landmarks on image
        # for i in range(landmarks.shape[0]):
        #     cv2.circle(image, (int(landmarks[i,0]), int(landmarks[i,1])), 1, (0,0,255), -1)
        #     # cv2.circle(masked_image, (int(landmarks[i,0]), int(landmarks[i,1])), 1, (0,0,255), -1)
        
        # # grid = np.concatenate([image, masked_image], axis=1)
        # cv2.imshow('grid', image)
        # cv2.waitKey(0)
