import os
import torch
import numpy as np
import torch.nn.functional as F
import cv2

def mask_generator(batch_size, input_size, mask_prob, block_size=16):
    
    if block_size is not None:
        # create a random binary mask with size block_size x block_size using torch.bernoulli
        mask = torch.bernoulli(mask_prob * torch.ones((batch_size, block_size, block_size))).cuda()
        
        # rescale the mask to size input_size x input_size
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=input_size, mode='nearest')
    else:
        mask = torch.bernoulli(mask_prob * torch.ones((batch_size, input_size, input_size))).unsqueeze(1).cuda()
    
    return mask

def load_templates():
    #templates_path = "assets/expression_templates"
    templates_path = "assets/exp_templates_famos_new_from_id"
    classes_to_load = ["lips_back", "rolling_lips", "mouth_side", "kissing", "high_smile", "mouth_up",
                       "mouth_middle", "mouth_down", "blow_cheeks", "cheeks_in", "jaw", "lips_up"]
    templates = {}
    for subject in os.listdir(templates_path):
        if os.path.isdir(os.path.join(templates_path, subject)):
            for template in os.listdir(os.path.join(templates_path, subject)):
                if template.endswith(".mp4"):
                    continue
                if template not in classes_to_load:
                    continue
                exps = []
                for npy_file in os.listdir(os.path.join(templates_path, subject, template)):
                    params = np.load(os.path.join(templates_path, subject, template, npy_file), allow_pickle=True)
                    exp = params.item()['expression'].squeeze()
                    exps.append(exp)
                templates[subject+template] = np.array(exps)
    print('Number of templates loaded: ', len(templates.keys()))
    # templates_path_spectre = "assets/exp_templates_spectre"
    
    
    # for template in os.listdir(templates_path_spectre):
    #     if template.endswith(".mp4"):
    #         continue
    #     exps = []
    #     params = torch.load(os.path.join(templates_path_spectre, template), map_location=torch.device('cpu'))
    #     self.templates[template] = params['exp'].numpy()[2:-2]

    
    return templates

def masking(img, mask, mask_ratio, wr=15, rendered_mask=None, diff_mask=None):
    # img: B x C x H x W
    # mask: B x 1 x H x W
    
    B, C, H, W = img.size()
    
    
    # save the first img in temp1.png
    #cv2.imwrite('temp1.png', cv2.cvtColor(img[0].permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
    
    # save the first mask in temp2.png
    #cv2.imwrite('temp2.png', mask[0].squeeze().cpu().numpy() * 255)
    
    mask = 1-F.max_pool2d(1-mask, 2 * wr + 1, stride=1, padding=wr)
    
    # save the first mask in temp3.png
    #cv2.imwrite('temp3.png', mask[0].squeeze().cpu().numpy() * 255)
    
    # find where rendered_img is black and replace it with the original image
    #rendered_background = (rendered_img == 0).all(dim=1, keepdim=True).float()


    # add random points inside the face 
    extra_points = mask_generator(img.size(0), H, mask_ratio, None)
    
    # save the extra_points in temp4.png
    #extra_points = F.max_pool2d(extra_points, 3, 1, 1)
    #cv2.imwrite('temp4.png', extra_points[0].squeeze().cpu().numpy() * 255)
    
    
    if rendered_mask is not None:
        extra_points = extra_points * rendered_mask
        #if diff_mask is not None:
        #    extra_points = extra_points * diff_mask
        #else:
        #   extra_points *= mask_generator(img.size(0), H, .5, 16)
        
        #cv2.imwrite('temp5.png', rendered_mask[0].squeeze().cpu().numpy() * 255)
        
        mask = torch.clip(mask * (1 - rendered_mask) + extra_points, 0, 1)
    else:
        mask = torch.clip(mask + extra_points, 0, 1)

    masked_img = img * mask
    masked_img = masked_img.detach()
    
    #cv2.imwrite('temp6.png', cv2.cvtColor(masked_img[0].permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_BGR2RGB))
    
    #raise Exception('stop')
    
    return masked_img


def mouth_mask(landmarks, image_size=(224, 224)):
    # landmarks: B x 68 x 2
    device = landmarks.device

    landmarks = landmarks.cpu().numpy()

    # normalize landmarks from [-1, 1] to image size
    landmarks = (landmarks + 1) * image_size[1] / 2

    B = landmarks.shape[0]
    # compute th convex hull of the mouth
    mouth_masks = np.zeros((B, image_size[0], image_size[1]))
    for i in range(B):
        mask = np.zeros(image_size)
        cv2.fillConvexPoly(mask, np.int32(landmarks[i, 60:68, :].round()), 1)
        mouth_masks[i] = mask

    return torch.from_numpy(mouth_masks).unsqueeze(1).float().to(device)



    