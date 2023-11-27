import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from skimage.transform import estimate_transform, warp
import albumentations as A
from datasets.lrs3_dataset import create_mask
import sys
from omegaconf import OmegaConf


class AffectNetDataset(Dataset):
    def __init__(self, data_list, config, test=False):
        self.data_list = data_list
        self.image_size = 224
        self.K = config.K
        self.test = test
        self.config=config
        
        # many color augmentations
        # + very small rotation & scale augmentations 
        self.transform = A.Compose([
                # color ones
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.25),
                A.CLAHE(p=0.255),
                #A.HueSaturationValue(p=0.25),  
                A.RGBShift(p=0.25),
                A.Blur(p=0.1),
                A.GaussNoise(p=0.5),
                # affine ones
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.9),
            ], keypoint_params=A.KeypointParams(format='xy'), additional_targets={'revmask': 'revmask'})
            
        if not self.test:
            self.scale = [1.4, 1.8]
        else:
            self.scale = 1.6

    def crop_face(self, frame, landmarks, scale=1.0):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):


        frame = cv2.imread(self.data_list[index][0])

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


        kpt_filename = self.data_list[index][1]
        if not os.path.exists(kpt_filename):
            return None
        
        kpt = np.load(kpt_filename, allow_pickle=True)

        if kpt is None or kpt.size == 1:
            return None
        
        kpt = kpt[0]

        if isinstance(self.scale, list):
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        else:
            scale = self.scale

        tform = self.crop_face(frame,kpt,scale)
        cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size), preserve_range=True).astype(np.uint8)
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
        cropped_kpt = cropped_kpt[:,:2]

        # find convex hull
        hull_mask = create_mask(cropped_kpt, (self.image_size, self.image_size), extend_to_forehead=True)
        # hull_mask_rev = 1 - hull_mask

        # augment
        if not self.test:
            transformed = self.transform(image=cropped_image, keypoints=cropped_kpt, mask= 1 - hull_mask)
            cropped_image = (transformed['image']/255.0).astype(np.float32)
            cropped_kpt = np.array(transformed['keypoints']).astype(np.float32)
            hull_mask = 1 - transformed['mask']
            # hull_mask_rev = transformed['revmask']
        else: 
            cropped_image = (cropped_image/255.0).astype(np.float32)


        cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1
        masked_cropped_image = cropped_image * hull_mask[...,None]

        cropped_image = cropped_image.transpose(2,0,1)
        masked_cropped_image = masked_cropped_image.transpose(2,0,1)

        images_array = torch.from_numpy(cropped_image).type(dtype = torch.float32) .unsqueeze(0) #1,3,224,224
        masked_images_array = torch.from_numpy(masked_cropped_image).type(dtype = torch.float32) .unsqueeze(0) #1,3,224,224

        kpt_array = torch.from_numpy(np.array(cropped_kpt)).type(dtype = torch.float32).unsqueeze(0) #K,224,224,3
        masks_array = torch.from_numpy(np.array(hull_mask[...,None])).type(dtype = torch.float32).unsqueeze(0) #K,224,224,3

        data_dict = {
            'img': images_array,
            'landmarks': kpt_array[...,:2],
            'masked_img': masked_images_array,
            'mask': masks_array,
        }



        return data_dict
    


    def __getitem__(self, index):
        landmarks_not_checked = True
        while landmarks_not_checked:
            data_dict = self.__getitem_aux__(index)
            # check if landmarks are not None
            if data_dict is not None:
                landmarks = data_dict['landmarks']
                if landmarks is not None and (landmarks.shape[-2] == 68):
                    landmarks_not_checked = False
                    break
            #else:
            print("Error in loading data. Trying again...")
            index = np.random.randint(0, len(self.data_list))

           
        
        return data_dict
def get_datasets_AffectNet(config=None):

    root = '/AffectNet-8Labels/'

    train_list = []
    for image in os.listdir(os.path.join(root,'train_set', 'images')):
        image_path = os.path.join(root,'train_set', 'images', image)
        landmark_path = os.path.join(root,'train_set', 'fan_landmarks', image.replace('.jpg', '.npy'))

        train_list.append([image_path, landmark_path])

    # val_list = []
    # for image in os.listdir(os.path.join(root,'val_set', 'images')):
    #     image_path = os.path.join(root,'val_set', 'images', image)
    #     landmark_path = os.path.join(root,'val_set', 'annotations', image.replace('.jpg', '_lnd.npy'))

    #     val_list.append([image_path, landmark_path])


    return AffectNetDataset(train_list, config) #, AffectNetDataset(val_list, config, test=True)    


if __name__ == '__main__':

    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    # Remove the configuration file name from sys.argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # merge config with cli args
    conf.merge_with_cli()

    train_dataset, val_dataset = get_datasets_AffectNet(conf)

    for i in range(100):
        sample = val_dataset[i]

        image = sample['frame']#.squeeze().numpy().transpose(1,2,0)
        # masked_image = sample['masked_img'].squeeze().numpy().transpose(1,2,0)
        landmarks = sample['kpt']#.squeeze().numpy()

        # plot landmarks on image
        for i in range(landmarks.shape[0]):
            # landmarks = landmarks * 112 + 112
            cv2.circle(image, (int(landmarks[i,0]), int(landmarks[i,1])), 1, (0,0,255), -1)
            # cv2.circle(masked_image, (int(landmarks[i,0]), int(landmarks[i,1])), 1, (0,0,255), -1)
        
        # grid = np.concatenate([image, masked_image], axis=1)
        cv2.imshow('grid', image)
        cv2.waitKey(0)
