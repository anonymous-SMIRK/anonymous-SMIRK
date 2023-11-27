import torch
import torch.nn.functional as F
from torch import nn
import timm
from transformers import MobileViTModel


class PoseEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        
        self.config = config
      
        if self.config.arch.backbone_pose == 'mobilevit':
            print('Using MobileViT in PoseEncoder')
            self.encoder = MobileViTModel.from_pretrained("apple/mobilevit-small")
            feature_size = 640
            #self.encoder.train()
        elif self.config.arch.backbone_pose == 'mobilenet':
            print('Using MobileNet in PoseEncoder')
            self.encoder = timm.create_model('tf_mobilenetv3_small_minimal_100', 
                            pretrained=True,
                            features_only=True,
                            )
            feature_size = 576
        elif self.config.arch.backbone_pose == 'resnet50':
            self.encoder = timm.create_model('resnet50',
                            pretrained=True,
                            features_only=True,
                            )
            feature_size = 2048
        else:
            raise NotImplementedError
        

        self.pose_cam_layers= nn.Sequential(
            nn.Linear(feature_size, 6)
        )

        #if config.resume is None:
        self.init_weights()


    def init_weights(self):
        self.pose_cam_layers[0].weight.data *= 0.001
        self.pose_cam_layers[0].bias.data *= 0.001

        self.pose_cam_layers[0].weight.data[...,3] = 0
        self.pose_cam_layers[0].bias.data[...,3] = 7


    def forward(self, img):
        # make B*T
        img_ = img#.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        if 'vit' in self.config.arch.backbone_pose:
            features = self.encoder(img_).last_hidden_state
        else:
            features = self.encoder(img_)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        outputs = {}

        pose_cam = self.pose_cam_layers(features).reshape(img_.size(0), -1)
        outputs['pose_params'] = pose_cam[...,:3]
        outputs['cam'] = pose_cam[...,3:]

        return outputs


class ShapeEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        if self.config.arch.backbone_shape == 'mobilevit':
            print('Using MobileViT in ShapeEncoder')

            self.encoder = MobileViTModel.from_pretrained("apple/mobilevit-small")
            feature_size = 640
            #self.encoder.train()
        elif self.config.arch.backbone_shape == 'mobilenet':
            print('Using MobileNet in ShapeEncoder')
            self.encoder = timm.create_model('tf_mobilenetv3_large_minimal_100', 
                            pretrained=True,
                            features_only=True,
                            )
            feature_size = 960
        elif self.config.arch.backbone_shape == 'resnet50':
            self.encoder = timm.create_model('resnet50',
                            pretrained=True,
                            features_only=True,
                            )
            feature_size = 2048
        else:
            raise NotImplementedError
        
        linear_size = config.arch.num_shape

        self.shape_layers = nn.Sequential(
            nn.Linear(feature_size, linear_size)
        )           

        #if config.resume is None:
        self.init_weights()


    def init_weights(self):

        self.shape_layers[0].weight.data *= 0
        self.shape_layers[0].bias.data *= 0


    def forward(self, img):
        img_ = img#.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        if 'vit' in self.config.arch.backbone_shape:
            features = self.encoder(img_).last_hidden_state
        else:
            features = self.encoder(img_)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


        parameters = self.shape_layers(features).reshape(img_.size(0), -1)

        return {'shape_params': parameters}


class ExpressionEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        
        self.config = config

        if self.config.arch.backbone_expression == 'mobilevit':
            print('Using MobileViT in ExpressionEncoder')

            self.encoder = MobileViTModel.from_pretrained("apple/mobilevit-small")
            feature_size = 640
            #self.encoder.train()
        elif self.config.arch.backbone_expression == 'mobilenet':
            print('Using MobileNet in ExpressionEncoder')
            self.encoder = timm.create_model('tf_mobilenetv3_large_minimal_100', 
                            pretrained=True,
                            features_only=True,
                            )
            feature_size = 960
        elif self.config.arch.backbone_expression == 'resnet50':
            self.encoder = timm.create_model('resnet50',
                            pretrained=True,
                            features_only=True,
                            )
            feature_size = 2048
        else:
            raise NotImplementedError
        
        linear_size = 50+2+3

        self.expression_layers = nn.Sequential(
            nn.Linear(feature_size, linear_size)
        )           

        #if config.resume is None:
        self.init_weights()


    def init_weights(self):

        self.expression_layers[0].weight.data *= 0
        self.expression_layers[0].bias.data *= 0


    def forward(self, img):
        img_ = img#.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

        if 'vit' in self.config.arch.backbone_expression:
            features = self.encoder(img_).last_hidden_state
        else:
            features = self.encoder(img_)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


        parameters = self.expression_layers(features).reshape(img_.size(0), -1)

        outputs = {}

        outputs['expression_params'] = parameters[...,:50]

        '''
        expression_params = parameters[...,:50]
        # decompose to magnitude and direction
        expression_params_magn = torch.mean(expression_params ** 2, dim=1, keepdim=True).sqrt()
        expression_params_dir = expression_params / (expression_params_magn.detach() + 1e-6)
        outputs['expression_params'] = torch.clamp(expression_params_magn, 0, 1.2) * expression_params_dir
        '''

        #outputs['eyelid_params'] = torch.clamp(parameters[...,50:52], 0, 1)
        # outputs['eyelid_params'] = parameters[...,50:52]
        outputs['eyelid_params'] = torch.clamp(parameters[...,50:52], 0, 1)

        #outputs['jaw_params'] = torch.zeros_like(parameters[...,52:55])
        #outputs['jaw_params'][...,0] = F.relu(parameters[...,52])
        #outputs['jaw_params'][...,1:3] = parameters[...,53:55]
        
        outputs['jaw_params'] = torch.cat([F.relu(parameters[...,52].unsqueeze(-1)), 
                                           torch.clamp(parameters[...,53:55], -.2, .2)], dim=-1)

        return outputs


class FLAME_Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()



        self.pose_encoder = PoseEncoder(config)

        self.shape_encoder = ShapeEncoder(config)

        self.expression_encoder = ExpressionEncoder(config) 

        self.config = config



    def forward(self, img):
        # make B*T

        pose_outputs = self.pose_encoder(img)
        shape_outputs = self.shape_encoder(img)
        expression_outputs = self.expression_encoder(img)

        outputs = {}
        outputs.update(pose_outputs)
        outputs.update(shape_outputs)
        outputs.update(expression_outputs)

        return outputs


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.encoder = FLAME_Encoder(config)
        self.config = config

    def forward(self, img):
        return self.encoder(img)


