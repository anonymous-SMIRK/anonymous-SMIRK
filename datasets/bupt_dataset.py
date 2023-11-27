import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from skimage.transform import estimate_transform, warp
import albumentations as A
from datasets.lrs3_dataset import create_mask
from datasets.base_dataset import BaseDataset

class BuptDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.keys = list(data_list.keys())
        
        # self.stable_points = np.array([19, 22, 25, 28, 16, 31, 37]) + 17

        # # self.stable_reference = np.load("assets/bupt_landmarks_mean.npy")
        # # self.stable_reference = self.stable_reference[self.stable_points]/200 * 2 - 1

        # self.stable_reference = np.load("assets/bupt_landmarks_mean.npy")
        # self.stable_reference = self.stable_reference/0.8
        # self.stable_reference[:,1] = self.stable_reference[:,1] + 0.15
        # self.stable_reference = self.stable_reference[self.stable_points]


    def __len__(self):
        return len(self.keys)

    def __getitem_aux__(self, index):
        images_list = []; kpt_list = []; masked_images_list = []; masks_list = [];

        key = self.keys[index]

        race = key.split('___')[0]
        subject = key.split('___')[1]

        landmarks_path = os.path.join(self.config.dataset.BUPT_landmarks_path, race, subject)
        folder_path = os.path.join(self.config.dataset.BUPT_path, race, subject)

        # files = sorted(os.listdir(folder_path))
        files = [x[2] for x in self.data_list[key]]
        # print(files)

        frame_indices = np.random.randint(0, len(files), size=self.K)

        if isinstance(self.scale, list):
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        else:
            scale = self.scale


        thetas = []
            
        for frame_idx in frame_indices:
            img_name = os.path.join(folder_path,files[frame_idx].replace('.npy','.jpg'))
            if not os.path.exists(img_name):
                img_name = os.path.join(folder_path,files[frame_idx].replace('.npy','.png'))

            frame = cv2.imread(img_name)

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            kpt_filename = os.path.join(landmarks_path,files[frame_idx])
            # if not os.path.exists(kpt_filename):
                # return None
            
            kpt = np.load(kpt_filename, allow_pickle=True)
            # print(kpt)
            if kpt is None or kpt.size == 1:
                print("no keypoints bupt")
                return None
            
            kpt = kpt[0]

            tform = self.crop_face(frame,kpt,scale)
            cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size), preserve_range=True).astype(np.uint8)
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T
            cropped_kpt = cropped_kpt[:,:2]


            # find convex hull
            hull_mask = create_mask(cropped_kpt, (self.image_size, self.image_size), extend_to_forehead=True)
            # hull_mask_rev = 1 - hull_mask

            # augment
            if not self.test:
                try:
                    transformed = self.transform(image=cropped_image, mask= 1 - hull_mask)
                    cropped_image = (transformed['image']/255.0).astype(np.float32)
                    # cropped_kpt = np.array(transformed['keypoints']).astype(np.float32)
                    hull_mask = 1 - transformed['mask']
                except ValueError: # this from albumentation
                    print("Error in albumentations...")
                    return None
                # hull_mask_rev = transformed['revmask']
            else: 
                cropped_image = (cropped_image/255.0).astype(np.float32)

        


            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            # landmarks_for_affine = cropped_kpt[self.stable_points]
            # H = estimate_transform('similarity', landmarks_for_affine[:,:2], self.stable_reference[:,:2])
            # theta = torch.inverse(torch.from_numpy(H.params)).float()[:2]


            masked_cropped_image = cropped_image * hull_mask[...,None]

            images_list.append(cropped_image.transpose(2,0,1))
            kpt_list.append(cropped_kpt)
            masked_images_list.append(masked_cropped_image.transpose(2,0,1))
            masks_list.append(hull_mask[...,None])
            # thetas.append(theta)


        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        masked_images_array = torch.from_numpy(np.array(masked_images_list)).type(dtype = torch.float32) #K,224,224,3
        try:
            kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        except:
            return None
        masks_array = torch.from_numpy(np.array(masks_list)).type(dtype = torch.float32)
        # thetas_array = torch.stack(thetas, dim=0)

        data_dict = {
            'img': images_array,
            'landmarks': kpt_array[...,:2],
            'masked_img': masked_images_array,
            'mask': masks_array,
        }

        # if self.config.arch.enable_stl:
            # data_dict['theta'] = thetas_array


        return data_dict
    



def get_datasets_BUPT(config=None):

    train_list = []

    # for race in ['Indian', 'African', 'Caucasian', 'Asian']:
    #     for subject in os.listdir(os.path.join(BUPT_path,race)):
    #         folder_path = os.path.join(BUPT_path, race, subject)
    #         landmarks_path = os.path.join(BUPT_landmarks_path, race, subject)

    #         include = True
    #         if os.path.exists(folder_path) and os.path.exists(landmarks_path):
    #             for kpt_filename in os.listdir(landmarks_path):
    #                 kpt = np.load(os.path.join(landmarks_path, kpt_filename), allow_pickle=True)
    #                 if kpt is None or kpt.size == 1:
    #                     include = False

    #         if include:
    #             train_list.append([folder_path, landmarks_path, subject])

    # print("BUPT dataset size: ", len(train_list))
    # np.save("bupt.npy", train_list)
    # train_dict = np.load('', allow_pickle=True)
    import pickle
    train_dict = pickle.load(open('datasets/bupt_train_list.pkl', 'rb'))
    # assert at least 4 frames per subject
    filtered_dict = {}
    for subject in train_dict.keys():
        if len(train_dict[subject]) >= config.K:
            filtered_dict[subject] = train_dict[subject]

    return BuptDataset(train_dict, config)



if __name__ == '__main__':

    # conf = OmegaConf.load(sys.argv[1])

    # OmegaConf.set_struct(conf, True)

    # Remove the configuration file name from sys.argv
    # sys.argv = [sys.argv[0]] + sys.argv[2:]

    # # merge config with cli args
    # conf.merge_with_cli()

    # train_dataset = get_datasets_BUPT(conf)

    # lds = []
    # from tqdm import tqdm
    # for i in tqdm(range(1000)):
    #     sample = train_dataset[i]
    #     lds.append(sample['landmarks'].squeeze().numpy())
    # lds = np.stack(lds, axis=0)
    # lds_mean = np.mean(lds, axis=0)

    # import matplotlib.pyplot as plt
    # plt.scatter(lds_mean[:,0], lds_mean[:,1])
    # # reverse axis
    # plt.gca().invert_yaxis()
    # plt.savefig("assets/bupt_mean.png")
    # np.save("assets/bupt_landmarks_mean.npy", lds_mean)
    


    # np.save("bupt_landmarks_mean.npy", lds)

    #     image = sample['img'].squeeze().numpy().transpose(1,2,0).copy()
    #     masked_image = sample['masked_img'].squeeze().numpy().transpose(1,2,0)
    #     landmarks = sample['landmarks'].squeeze().numpy()

    #     landmarks = landmarks * 112 + 112
    #     # # plot landmarks on image
    #     for i in range(landmarks.shape[0]):
    #         cv2.circle(image, (int(landmarks[i,0]), int(landmarks[i,1])), 1, (0,0,255), -1)
    #         # cv2.circle(masked_image, (int(landmarks[i,0]), int(landmarks[i,1])), 1, (0,0,255), -1)
        
    #     # grid = np.concatenate([image, masked_image], axis=1)
    #     cv2.imshow('grid', image)
    #     cv2.waitKey(0)
    base = np.load("assets/bupt_landmarks_mean.npy")
    base = base/0.8
    base[:,1] = base[:,1] + 0.15
    import matplotlib.pyplot as plt
    plt.scatter(base[:,0], base[:,1])
    # stable_points = np.array([19, 22, 25, 28, 16, 31, 37]) + 17

    # stable_reference = np.load("assets/20words_mean_face.npy")
    # # stable_reference = stable_reference/200 * 2 - 1
    # import matplotlib.pyplot as plt
    # plt.scatter(stable_reference[:,0], stable_reference[:,1])
    # # reverse axis
    plt.gca().invert_yaxis()
    plt.savefig("tmp.png")
    
