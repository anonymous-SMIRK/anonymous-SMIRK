import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from models.FLAME.FLAME import FLAME, FLAMETex
from models.FLAME.util import batch_orth_proj
from models.FLAME.deca_renderer import SRenderY

#from models_new.encoder_all import Encoder
from models.encoder import Encoder
from models.encoder_deca import DECA_Encoder
from models.unet import UNet
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import cv2
import os
import random
import copy


from models.utils import *

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()


        self.config = config

        if config.train.loss_weights['perceptual_loss'] > 0 or config.train.loss_weights['reconstruction_loss'] > 0 \
            or config.train.loss_weights['emotion_loss'] > 0 or config.train.loss_weights['lipreading_loss'] > 0 or config.train.loss_weights['identity_loss'] > 0:
            print('Enabling fuse generator')
            self.enable_fuse_generator = True
        else:
            print('Disabling fuse generator')
            self.enable_fuse_generator = False

        if self.enable_fuse_generator:
            self.fuse_generator = UNet(in_channels=6, out_channels=3, init_features=32, res_blocks=5)

        self.encoder = Encoder(config)

        if config.train.use_base_model_for_regularization:
            self.base_encoder = copy.deepcopy(self.encoder)
        
        self.deca_encoder = DECA_Encoder()
        for param in self.deca_encoder.parameters():
            param.requires_grad_(False)
        self.deca_encoder.eval()

        self.flame = FLAME(n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)

        self.renderer = SRenderY(config) #.cuda()
        self.setup_losses()

        self.logger = None

        if self.config.arch.use_expression_templates_in_second_path:
            self.templates = load_templates()
            
    def load_random_template(self, num_expressions=50):
        
        # select a random key from self.templates
        random_key = random.choice(list(self.templates.keys()))
        
        # load random template
        templates = self.templates[random_key]
        
        # select a random index from the first dimension
        random_index = random.randint(0, templates.shape[0]-1)
        
        return templates[random_index][:num_expressions]
        
        
    def setup_logger(self, wandb_run):
        self.logger = wandb_run

    def setup_losses(self):
        from models.losses.VGGPerceptualLoss import VGGPerceptualLoss
        self.vgg_loss = VGGPerceptualLoss()
        self.vgg_loss.eval()
        for param in self.vgg_loss.parameters():
            param.requires_grad_(False)
        
        if self.config.train.loss_weights['identity_loss'] > 0:
            from models.losses.VGGFace2Loss import VGGFace2Loss
            self.identity_loss = VGGFace2Loss()
            # freeze the identity model
            self.identity_loss.eval()
            #for param in self.identity_loss.parameters():
            #    param.requires_grad_(False)
            
        if self.config.train.loss_weights['emotion_loss'] > 0:
            from models.losses.ExpressionLoss import ExpressionLoss
            self.emotion_loss = ExpressionLoss()
            # freeze the emotion model
            self.emotion_loss.eval()
            for param in self.emotion_loss.parameters():
                param.requires_grad_(False)

        if self.config.train.loss_weights['mica_loss'] > 0:
            from models.MICA.mica import MICA

            self.mica = MICA()
            self.mica.eval()

            for param in self.mica.parameters():
                param.requires_grad_(False)

    def eval(self):
        self.encoder.eval()
        if self.enable_fuse_generator:
            self.fuse_generator.eval()
            
    def train(self):
        self.encoder.train()
        if self.enable_fuse_generator:
            self.fuse_generator.train()
            
    def preprocess_batch(self, batch):
        
        #B, T, C, H, W = batch['img'].shape
        
        kmask = batch['K_useful']
        
        for key in batch.keys():
            if key == 'K_useful' or key == 'subject' or key == 'filename':
                continue
            if key == 'dataset_name':
                # list of strings
                if sum(kmask) > 0:
                    #print(np.asarray(batch[key]), kmask.numpy())
                    list1 = [t for t in np.asarray(batch[key])[kmask.numpy()] for _ in range(self.config.K)]
                else:
                    list1 = []
                if sum(~kmask) > 0:
                    list2 = [t for t in np.asarray(batch[key])[~kmask.numpy()]]
                else:
                    list2 = []
                batch[key] =    list1 + list2
                continue
                
            # per batch flag
            if key == 'flag_landmarks_fan':
                batch[key] = torch.cat([
                    batch[key][kmask].view(-1, 1).repeat(1, self.config.K).view(-1),
                    batch[key][~kmask].view(-1)
                ], dim=0)
            else:
                tsize = batch[key].shape[2:]
                batch[key] = torch.cat([
                    batch[key][kmask].view(-1, *tsize),
                    batch[key][~kmask][:, 0].view(-1, *tsize)   
                ], dim=0)
            
            batch[key] = batch[key].cuda()  
            #print(key, batch[key].shape)

        # number of K-tuples in the batch
        batch['NK'] = torch.sum(kmask).item() * self.config.K
        
        return batch
        

    def step1(self, batch, batch_idx, phase='train'):

        #B, T, C, H, W = batch['img'].shape
        C = batch['img'].shape[1]

        rmask = 1 #mask_generator(B, H, .9, 56).unsqueeze(1)
        encoder_output = self.encoder(batch['img'] * rmask)
        
        with torch.no_grad():
            base_output = self.base_encoder(batch['img'])

        flame_output = self.forward_flame(encoder_output, use_eyelids=self.config.arch.use_eyelids)
        

        rendered_img = self.render(flame_output['verts'], flame_output['trans_verts'])

        if phase == 'test':
            flame_output_zero = self.forward_flame(encoder_output, use_eyelids=False, zero_expression=True)
            flame_output = self.forward_flame(encoder_output, use_eyelids=True, zero_expression=False)

            rendered_img_zero = self.render(flame_output_zero['verts'], flame_output_zero['trans_verts'])
            rendered_img = self.render(flame_output['verts'], flame_output['trans_verts'])

            outputs = {
                'img': batch['img'].reshape(-1, 3, self.config.image_size, self.config.image_size),
                'rendered_img': rendered_img,
                'rendered_img_zero': rendered_img_zero,
                'landmarks2d': flame_output['landmarks2d'],
                'verts_zero': flame_output_zero['verts'],
                'verts': flame_output['verts']
            }
            if 'landmarks_fan' in batch.keys():
                outputs['landmarks_fan'] = batch['landmarks_fan']#.reshape(-1, batch_1_flattened['landmarks_fan'].shape[2], batch_1_flattened['landmarks_fan'].shape[3])
            if 'landmarks_mp' in batch.keys():
                outputs['landmarksgt_mp'] = batch['landmarks_mp']#.reshape(-1, batch_1_flattened['landmarks_mp'].shape[2], batch_1_flattened['landmarks_mp'].shape[3])
             
            for key in outputs.keys():
                outputs[key] = outputs[key].detach().cpu()

            return outputs


        # ---------------- losses ---------------- #
        losses = {}
        # torch.autograd.set_detect_anomaly(True)
        # -------- flatten img and landmarks to B*T x C x H x W -------- #
        img = batch['img']#.reshape(-1, C, H, W)
        landmarks2d = batch['landmarks_fan']#.reshape(-1, batch['landmarks_fan'].shape[2], batch['landmarks_fan'].shape[3])
        landmarks_mp = batch['landmarks_mp']#.reshape(-1, batch['landmarks_mp'].shape[2], batch['landmarks_mp'].shape[3])

        valid_landmarks = batch['flag_landmarks_fan']

        if torch.sum(valid_landmarks) == 0:
            # if there are no valid landmarks then skip this batch
            losses['landmark_loss'] = 0
        else:
            losses['landmark_loss'] = F.mse_loss(flame_output['landmarks2d'][valid_landmarks,:17], landmarks2d[valid_landmarks,:17]) # get only the first 17 landmarks for boundary
    
        # print(flame_output['landmarksmp'], landmarks_mp)
        losses['landmark_loss_mp'] = F.mse_loss(flame_output['landmarksmp'], landmarks_mp)

        #  ---------------- regularization losses ---------------- # 
        if self.config.train.use_base_model_for_regularization:
            # expression regularization
            losses['expression_regularization'] = torch.mean((encoder_output['expression_params'] - base_output['expression_params'])**2)

            # shape regularization
            losses['shape_regularization'] = 1.0 * torch.mean((encoder_output['shape_params'][..., :35] - base_output['shape_params'][..., :35])**2) + \
                                                   torch.mean((encoder_output['shape_params'][..., 35:] - base_output['shape_params'][..., 35:])**2)

            # jaw regularization
            losses['jaw_regularization'] = torch.mean((encoder_output['jaw_params'] - base_output['jaw_params'])**2)
        else:
            # expression regularization
            losses['expression_regularization'] = torch.mean(encoder_output['expression_params']**2)

            # shape regularization
            losses['shape_regularization'] = 1.0 * torch.mean(encoder_output['shape_params'][..., :35]**2) + \
                                                   torch.mean(encoder_output['shape_params'][..., 35:]**2)

            # jaw regularization
            losses['jaw_regularization'] = torch.mean(encoder_output['jaw_params']**2)



        if self.enable_fuse_generator:
            masks = batch['mask']#.reshape(-1, 1, H, W)

            # mask out face and add random points inside the face
            rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
            masked_img = masking(img, masks, self.config.train.mask_ratio, self.config.train.mask_dilation_radius, rendered_mask=rendered_mask)
            reconstructed_img = self.fuse_generator(torch.cat([rendered_img, masked_img], dim=1))
        
            # reconstruction loss
            #img_mean, img_std = torch.mean(img, dim=[2,3], keepdim=True).detach(), torch.std(img, dim=[2,3], keepdim=True).detach() + 1e-5
            img_mean, img_std = 0, 1
            reconstruction_loss = F.l1_loss((reconstructed_img - img_mean)/img_std, (img - img_mean)/img_std, reduction='none')
            # for visualization
            loss_img = reconstruction_loss.mean(dim=1, keepdim=True)
            losses['reconstruction_loss'] = reconstruction_loss.mean()

            # perceptual loss
            losses['perceptual_loss'] = self.vgg_loss(reconstructed_img, img)
            #losses['perceptual_loss'] = self.identity_loss(reconstructed_img, img, metric='l1')


            # perceptual losses
            perceptual_losses = 0
            if self.config.train.loss_weights['lipreading_loss'] > 0 or self.config.train.loss_weights['emotion_loss'] > 0:
                # do not let this gradient flow through the generator
                for param in self.fuse_generator.parameters():
                    param.requires_grad_(False)
                self.fuse_generator.eval()
                reconstructed_img_p = self.fuse_generator(torch.cat([rendered_img, masked_img], dim=1))
                for param in self.fuse_generator.parameters():
                    param.requires_grad_(True)
                self.fuse_generator.train()

            # emotion loss
            if self.config.train.loss_weights['emotion_loss'] > 0:
                mead_mask = torch.Tensor(['mead' in tname.lower() for tname in batch['dataset_name']]).bool()
               
                valid_mask = mead_mask.to(img.device)
                if torch.sum(valid_mask) == 0:
                    losses['emotion_loss'] = 0
                else:
                    losses['emotion_loss'] = self.emotion_loss(reconstructed_img_p[valid_mask], img[valid_mask], metric='l2', use_mean=False)
                    losses['emotion_loss'] = losses['emotion_loss'].mean()
                perceptual_losses += losses['emotion_loss'] * self.config.train.loss_weights['emotion_loss']

        # mica loss
        if self.config.train.loss_weights['mica_loss'] > 0:
            with torch.no_grad():
                mica_output = self.mica(batch['img_mica'].reshape(-1, 3, 112, 112))
            encoder_shape = encoder_output['shape_params']
            if encoder_shape.size(-1) < mica_output['shape_params'].size(-1): 
                encoder_shape = torch.cat([encoder_shape, torch.zeros(encoder_shape.size(0), mica_output['shape_params'].size(-1) - encoder_shape.size(-1)).cuda()], dim=-1)
            losses['mica_loss'] = F.mse_loss(encoder_shape[valid_landmarks], mica_output['shape_params'][valid_landmarks])
        else:
            losses['mica_loss'] = 0
            
        if self.config.train.loss_weights['deca_loss'] > 0:
            with torch.no_grad():
                deca_output = self.deca_encoder(batch['img'])
            deca_shape = deca_output['shape_params'].detach()
            if encoder_output['shape_params'].size(-1) > deca_shape.size(-1):
                deca_shape = torch.cat([deca_shape, torch.zeros(deca_shape.size(0), encoder_output['shape_params'].size(-1) - deca_shape.size(-1)).cuda()], dim=-1)
            losses['deca_loss'] = F.mse_loss(encoder_output['shape_params'], deca_shape)
        else:
            losses['deca_loss'] = 0

        
        # shape consistency loss to be added for shape

        reg_loss =  losses['expression_regularization'] * self.reg_struct['expression_regularization'] + \
                    losses['shape_regularization'] * self.reg_struct['shape_regularization'] + \
                    losses['jaw_regularization'] * self.reg_struct['jaw_regularization'] 
                    
        shape_losses =      losses['mica_loss'] * self.config.train.loss_weights['mica_loss'] + \
                            losses['deca_loss'] * self.config.train.loss_weights['deca_loss']

        loss_first_path =   losses['landmark_loss'] * self.config.train.loss_weights['landmark_loss'] + \
                            losses['landmark_loss_mp'] * self.config.train.loss_weights['landmark_loss'] + \
                            reg_loss + \
                            shape_losses

        if self.enable_fuse_generator:        
            fuse_generator_losses = losses['perceptual_loss'] * self.config.train.loss_weights['perceptual_loss'] + \
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss'] + \
                                perceptual_losses 
            loss_first_path += fuse_generator_losses
        

        # ---------------- optimize first path ---------------- #
        if phase == 'train':
            # clear gradients
            if self.enable_fuse_generator:
                self.fuse_generator.zero_grad()
    
            self.encoder_optimizer.zero_grad()
            loss_first_path.backward()
            self.encoder_optimizer.step()

            if self.enable_fuse_generator:
                self.fuse_generator_optimizer.step()
        
        
        # ---------------- visualization struct ---------------- #
        outputs = {}
        if batch_idx % self.config.train.visualize_every == 0:
            with torch.no_grad():

                deca_output = self.deca_encoder(batch['img'])
                flame_output_deca = self.forward_flame(deca_output, use_eyelids=False)
                rendered_img_deca = self.render(flame_output_deca['verts'], flame_output_deca['trans_verts'])
                outputs['rendered_img_deca'] = rendered_img_deca.detach().cpu()
                
                flame_output_deca_zero = self.forward_flame(deca_output, use_eyelids=False, zero_expression=True)
                rendered_img_deca_zero = self.render(flame_output_deca_zero['verts'], flame_output_deca_zero['trans_verts'])
                outputs['rendered_img_deca_zero'] = rendered_img_deca_zero.detach().cpu()

                flame_output_base = self.forward_flame(base_output, use_eyelids=True)
                rendered_img_base = self.render(flame_output_base['verts'], flame_output_base['trans_verts'])
                outputs['rendered_img_base'] = rendered_img_base.detach().cpu()
            

                #deca_output = self.deca_encoder(batch['img'])
                outputs['rendered_img'] = rendered_img.detach().cpu()
            
                flame_output_zero = self.forward_flame(encoder_output, use_eyelids=False, zero_expression=True)
                rendered_img_zero = self.render(flame_output_zero['verts'], flame_output_zero['trans_verts'])
                outputs['rendered_img_zero'] = rendered_img_zero.detach().cpu()
           
                outputs['img'] = img.detach().cpu() #* (1 - batch['mask'].reshape(-1, 1, H, W))
                outputs['landmarksgt'] = landmarks2d.detach().cpu()
                outputs['landmarksgt_mp'] = landmarks_mp.detach().cpu()
                outputs['landmarks2d'] = flame_output['landmarks2d'].detach().cpu()
                outputs['landmarksmp'] = flame_output['landmarksmp'].detach().cpu()

                if self.enable_fuse_generator:
                    outputs['loss_img'] = loss_img.detach().cpu()
                    outputs['reconstructed_img'] = reconstructed_img.detach().cpu()
                    outputs['masked_1st_path'] = masked_img.detach().cpu()

                
                if self.config.train.loss_weights['mica_loss'] > 0:
                    mica_encoder_output = {}
                    for key in encoder_output.keys():
                        mica_encoder_output[key] = encoder_output[key].clone().detach()
                    mica_encoder_output['shape_params'] = mica_output['shape_params']

                    mica_output_zero = self.forward_flame(mica_encoder_output, use_eyelids=False, zero_expression=True)
                    rendered_img_mica_zero = self.render(mica_output_zero['verts'], mica_output_zero['trans_verts'])

                    outputs['rendered_img_mica_zero'] = rendered_img_mica_zero.detach().cpu()
                    outputs['img_mica'] = batch['img_mica'].reshape(-1, C, 112, 112)
                    outputs['img_mica'] = F.interpolate(outputs['img_mica'], self.config.image_size).detach().cpu()
              
        return outputs, losses, encoder_output
        
    # ---------------- second path ---------------- #
    def step2(self, encoder_output, batch, batch_idx, phase='train'):
        
        B, C, H, W = batch['img'].shape
        
        img = batch['img'] # .reshape(-1, C, H, W)
        masks = batch['mask'] #.reshape(-1, 1, H, W)

        if self.config.arch.freeze_generator_in_second_path:
            self.fuse_generator.eval()
            for param in self.fuse_generator.parameters():
                param.requires_grad_(False)

        losses = {}

        # number of multiple versions for the second path
        # hardcoded for now!
        if self.config.arch.use_expression_templates_in_second_path:
            Ke = 4
        else:
            Ke = 3 #self.config.train.Ke 

        # start from the same encoder output and add noise to expression params
        # hard clone flame_feats
        flame_feats = {}
        for k, v in encoder_output.items():

            tmp = v.clone().detach()
            flame_feats[k] = torch.cat(Ke * [tmp], dim=0)


        expression_params_size = (B, flame_feats['expression_params'].size(1))
        # ---------------- random expression ---------------- #
        # 1 of 4 Ke - random expressions!        
        
        param_mask = torch.bernoulli(torch.ones(expression_params_size) * 0.5).cuda()
        #new_expressions = 2 * (torch.rand(flame_feats['expression_params'].size()).cuda()-1) * 2.0 * param_mask + flame_feats['expression_params']
        new_expressions = (torch.randn(expression_params_size).cuda()) * 2.0 * param_mask + flame_feats['expression_params'][:B]
        
        #print((new_expressions ** 2).mean(dim=1) )

        #expression_diff = new_expressions - flame_feats['expression_params'][:B]
        # maximum magnitude of expression difference should be 2.5 
        #expression_diff_magn = (expression_diff ** 2).mean(dim=1) 
        #expresion_scale = (torch.clamp(expression_diff_magn, 0, 2.0) / expression_diff_magn).sqrt()
        expression_magn = (new_expressions ** 2).mean(dim=1) 
        expresion_scale = (torch.clamp(expression_magn, 0, 2.0) / expression_magn).sqrt()
        flame_feats['expression_params'][:B] = expresion_scale.view(-1, 1) * new_expressions
        
        # ---------------- permutation of expression ---------------- #
        # 2 of 4 Ke - permutation!     
        flame_feats['expression_params'][B: 2 * B] = flame_feats['expression_params'][B: 2 * B][torch.randperm(B)] + .25 * torch.randn(expression_params_size).cuda()
        
        # ---------------- template injection ---------------- #
        # 3 of 4 Ke - template injection!   
        
        if self.config.arch.use_expression_templates_in_second_path:
            for i in range(B):
                expression = self.load_random_template(num_expressions=self.config.arch.num_expression)
                flame_feats['expression_params'][2 * B + i,:self.config.arch.num_expression] = torch.Tensor(expression).cuda()
        flame_feats['expression_params'][2 * B: 3 * B] += .25 * torch.randn(expression_params_size).cuda()


        # ---------------- tweak jaw for all paths ---------------- #
        scale_mask = torch.Tensor([1, .1, .1]).cuda().view(1, 3) * torch.bernoulli(torch.ones(Ke * B) * 0.5).cuda().view(-1, 1)
        flame_feats['jaw_params'] = flame_feats['jaw_params']  + torch.randn(flame_feats['jaw_params'].size()).cuda() * 0.2 * scale_mask
        flame_feats['jaw_params'][..., 0] = torch.clamp(flame_feats['jaw_params'][..., 0] , 0.0, 0.5)
        
        # ---------------- tweak eyelids for all paths ---------------- #
        if self.config.arch.use_eyelids:
            if random.random() < 0.8:
                flame_feats['eyelid_params'] += (-1 + 2 * torch.rand(size=flame_feats['eyelid_params'].size()).cuda()) * 0.2
                # flame_feats['eyelid_params'] = torch.clamp(flame_feats['eyelid_params'], -0.5, 1.0)
                flame_feats['eyelid_params'] = torch.clamp(flame_feats['eyelid_params'], 0.0, 1.0)

        # ---------------- zero expression ---------------- #
        # 4 of 4 Ke - zero expression!     
        # use zero expression as one of the paths if Ke > 1 - let the eyelids to move a lot
        if Ke > 1:
            flame_feats['expression_params'][-B:] *= 0.0
            flame_feats['jaw_params'][-B:] *= 0.0
            flame_feats['eyelid_params'][-B:] = torch.rand(size=flame_feats['eyelid_params'][-B:].size()).cuda()        

        flame_feats['expression_params'] = flame_feats['expression_params'].detach()
        flame_feats['pose_params'] = flame_feats['pose_params'].detach()
        flame_feats['shape_params'] = flame_feats['shape_params'].detach()
        flame_feats['jaw_params'] = flame_feats['jaw_params'].detach()
        flame_feats['eyelid_params'] = flame_feats['eyelid_params'].detach()

        # render the new mesh
        flame_output_2nd_path = self.forward_flame(flame_feats, use_eyelids=self.config.arch.use_eyelids)
        rendered_img_2nd_path = self.render(flame_output_2nd_path['verts'], flame_output_2nd_path['trans_verts']).detach() # detach!
        
        
        rendered_mask = (rendered_img_2nd_path > 0).all(dim=1, keepdim=True).float()
        masked_img_2nd_path = masking(img.repeat(Ke, 1, 1, 1), masks.repeat(Ke, 1, 1, 1), self.config.train.mask_ratio, self.config.train.mask_dilation_radius, 
                                      rendered_mask=rendered_mask)
        reconstructed_img_2nd_path = self.fuse_generator(torch.cat([rendered_img_2nd_path, masked_img_2nd_path], dim=1))
        
        rmask = 1 #mask_generator(Ke * B, H, .9, 56).unsqueeze(1)
        recon_feats = self.encoder(reconstructed_img_2nd_path.view(Ke * B, C, H, W) * rmask) 


        flame_output_2nd_path_2 = self.forward_flame(recon_feats, use_eyelids=self.config.arch.use_eyelids)
        rendered_img_2nd_path_2 = self.render(flame_output_2nd_path_2['verts'], flame_output_2nd_path_2['trans_verts'])

        cycle_loss =    1.0 * F.mse_loss(recon_feats['expression_params'], flame_feats['expression_params']) + \
                        1.0  * F.mse_loss(recon_feats['jaw_params'], flame_feats['jaw_params']) + \
                        1.0 * F.mse_loss(recon_feats['shape_params'], flame_feats['shape_params']) 

        if self.config.arch.use_eyelids:
            cycle_loss += F.mse_loss(recon_feats['eyelid_params'], flame_feats['eyelid_params'])
        
        losses['cycle_loss']  = cycle_loss
        loss_second_path = losses['cycle_loss'] * self.cycle_loss_weight

        # identity loss
        if self.config.train.loss_weights['identity_loss'] > 0:
            # freeze the generator
            identity_loss = self.identity_loss(reconstructed_img_2nd_path, img.repeat(Ke, 1, 1, 1), use_mean=False)
            losses['identity_loss'] = F.relu(identity_loss - 0.3).mean()
            
            loss_second_path +=  losses['identity_loss'] * self.config.train.loss_weights['identity_loss'] 

        # ---------------- optimize second path ---------------- #
        if phase == 'train':
            # clear gradients
            if not self.config.arch.freeze_generator_in_second_path:
                self.fuse_generator_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            loss_second_path.backward()
            if not self.config.arch.freeze_generator_in_second_path:
                self.fuse_generator_optimizer.step()
            self.encoder_optimizer.step()
            
            
        if self.config.arch.freeze_generator_in_second_path:
            self.fuse_generator.train()
            for param in self.fuse_generator.parameters():
                param.requires_grad_(True)
            
        # ---------------- visualization struct ---------------- #
        
        outputs = {}
        if batch_idx % self.config.train.visualize_every == 0:
            
            outputs['2nd_path'] = torch.stack([rendered_img_2nd_path.detach().cpu().view(Ke, B,  C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W), 
                                             reconstructed_img_2nd_path.detach().cpu().view(Ke, B,  C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W), 
                                             rendered_img_2nd_path_2.detach().cpu().view(Ke, B, C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W)], dim=1).reshape(-1, C, H, W)

        return outputs, losses
        
    def logging(self, batch_idx, losses, phase):

        # ---------------- logging ---------------- #
        if self.config.train.log_losses_every > 0 and batch_idx % self.config.train.log_losses_every == 0:
            # print losses in one line
            loss_str = ''
            for k, v in losses.items():
                loss_str += f'{k}: {v:.6f} '
                if self.logger is not None:
                    self.logger.log({f'{phase}/{k}': v})
            print(loss_str)


    def step(self, batch, batch_idx, phase='train'):

        #print('second path: ', self.second_path)

        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)
        
        # do not compute pose
        if not self.config.arch.optimize_pose:
            #print('Freezing pose encoder')
            self.encoder.encoder.pose_encoder.eval()
            # do now allow gradients to flow through the pose encoder
            for param in self.encoder.encoder.pose_encoder.parameters():
                param.requires_grad = False
        
        if not self.config.arch.optimize_shape:
            #print('Freezing shape encoder')
            self.encoder.encoder.shape_encoder.eval()
            for param in self.encoder.encoder.shape_encoder.parameters():
                param.requires_grad = False
        
        batch = self.preprocess_batch(batch)
        
        # main calls
        outputs1, losses1, encoder_output = self.step1(batch, batch_idx, phase)
        if (self.config.train.loss_weights['cycle_loss'] > 0) and (phase == 'train'):
            outputs2, losses2 = self.step2(encoder_output, batch, batch_idx, phase)
        else:
            outputs2, losses2 = {}, {}
            
            
        # logging 
        losses = {**losses1, **losses2}
        
        self.logging(batch_idx, losses, phase)
        
        # for visualization
        outputs = {**outputs1, **outputs2}
     
        self.scheduler_step()

        return outputs 

    def configure_optimizers(self, n_steps):
        
        self.n_steps = n_steps
        encoder_scale = .5

        # check if self.encoder_optimizer exists
        if hasattr(self, 'encoder_optimizer'):
            for g in self.encoder_optimizer.param_groups:
                g['lr'] = encoder_scale * self.config.train.lr
        else:
            params = list(self.encoder.encoder.expression_encoder.parameters()) 
            if self.config.arch.optimize_shape:
                params += list(self.encoder.encoder.shape_encoder.parameters())
            if self.config.arch.optimize_pose:
                params += list(self.encoder.encoder.pose_encoder.parameters())
            self.encoder_optimizer = torch.optim.Adam(params, lr= encoder_scale * self.config.train.lr)
                
        # cosine schedulers for both optimizers - per iterations
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=n_steps,
                                                                                  eta_min=0.01 * encoder_scale * self.config.train.lr)
        if self.enable_fuse_generator:
            if hasattr(self, 'fuse_generator_optimizer'):
                for g in self.fuse_generator_optimizer.param_groups:
                    g['lr'] = self.config.train.lr
            else:
                self.fuse_generator_optimizer = torch.optim.Adam(self.fuse_generator.parameters(), lr= self.config.train.lr, betas=(0.5, 0.999))

            
            self.fuse_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.fuse_generator_optimizer, T_max=n_steps,
                                                                                    eta_min=0.01 * self.config.train.lr)        

        
    def scheduler_step(self):
        self.encoder_scheduler.step()
        if self.enable_fuse_generator:
            self.fuse_generator_scheduler.step()
        


    def forward_flame(self, encoder_output, use_eyelids, zero_expression=False, zero_pose=False, zero_shape=False):
        expression_params = encoder_output['expression_params']
        shape_params = encoder_output['shape_params']
        jaw_params = encoder_output['jaw_params']
        pose_params = encoder_output['pose_params']
        cam = encoder_output['cam']
        eyelid_params = None

        # Adjust expression params size if needed
        if expression_params.shape[1] < self.config.arch.num_expression:
            expression_params = torch.cat([expression_params, torch.zeros(expression_params.shape[0], self.config.arch.num_expression - expression_params.shape[1]).cuda()], dim=1)

        if shape_params.shape[1] < self.config.arch.num_shape:
            shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.config.arch.num_shape - shape_params.shape[1]).cuda()], dim=1)


        # Set eyelid params if used
        if use_eyelids:
            eyelid_params = encoder_output['eyelid_params']

        
        # Zero out the expression and pose parameters if needed
        if zero_expression:
            expression_params = torch.zeros_like(expression_params).cuda()
            jaw_params = torch.zeros_like(jaw_params).cuda()

        if zero_pose:
            pose_params = torch.zeros_like(pose_params).cuda()
            # if not zero_pose:
            pose_params[...,0] = 0.2
            pose_params[...,1] = -0.7

            cam = torch.zeros_like(cam).cuda()
            cam[...,0] = 7

        if zero_shape:
            shape_params = torch.zeros_like(shape_params).cuda()


        # Call to flame.forward with appropriate parameters
        verts, landmarks2d, _, landmarksmp = self.flame.forward(expression_params=expression_params,
                                                shape_params=shape_params,
                                                jaw_params=jaw_params,
                                                pose_params=pose_params,
                                                eyelid_params=eyelid_params)

        
        trans_verts = batch_orth_proj(verts, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        landmarks2d = batch_orth_proj(landmarks2d, cam)
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
        landmarks2d = landmarks2d[:, :, :2]

        landmarksmp = batch_orth_proj(landmarksmp, cam)
        landmarksmp[:, :, 1:] = -landmarksmp[:, :, 1:]
        landmarksmp = landmarksmp[:, :, :2]

        outputs = { 
                'verts': verts,
                'trans_verts': trans_verts,
                'landmarks2d': landmarks2d,
                'landmarksmp': landmarksmp
            }
        
        return outputs


    def render(self, verts, trans_verts):
        
        rendered_img = self.renderer.forward(verts, trans_verts)

        return rendered_img


    def visualize(self, outputs, epoch, it, save_path, show_landmarks=True):
        nrow = 1
        
        if 'img' in outputs and 'rendered_img' in outputs:
            outputs['overlap_image'] = outputs['img'] * 0.7 + outputs['rendered_img'] * 0.3


        original_img_with_landmarks = outputs['img']
        original_grid = make_grid(original_img_with_landmarks, nrow=nrow)


        image_keys = ['img_mica', 
                      'rendered_img_deca', 'rendered_img_base', 'rendered_img_emoca', 'rendered_img_pretrained', 'rendered_img', 'overlap_image', 
                    #   'rendered_img_deca_zero', 'rendered_img_mica_zero', 'rendered_img_zero', 'rendered_img_coma', 
                      'rendered_img_deca_zero', 'rendered_img_emoca_zero', 'rendered_img_pretrained_zero', 'rendered_img_mica_zero', 'rendered_img_zero', 'rendered_img_coma', 
                      'error_img_deca', 'error_img_emoca', 'error_img_pretrained', 'error_img',
                      'masked_1st_path', 'reconstructed_img', 'loss_img', 
                      '2nd_path']
                      #'rendered_img_2nd_path', 'reconstructed_img_2nd_path', 'rendered_img_2nd_path_2']
        
        nrows = [1 if '2nd_path' not in key else 3 * self.config.train.Ke for key in image_keys]
        
        grid = torch.cat([original_grid] + [make_grid(outputs[key], nrow=nr) for key, nr in zip(image_keys, nrows) if key in outputs.keys()], dim=2)
            
        grid = grid.permute(1,2,0).cpu().numpy()*255.0
        # clamp
        grid = np.clip(grid, 0, 255)
        grid = grid.astype(np.uint8)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        name = f'{epoch}_{it}.jpg'

        cv2.imwrite(os.path.join(save_path, name), grid)
