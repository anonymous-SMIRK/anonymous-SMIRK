import torch
import torch.nn.functional as F
from torch import nn
from models.resnet import load_ResNet50Model

class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()

        self.feature_size = 2048
        self.outsize = 236

        self.encoder = load_ResNet50Model() 

        self.layers = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.outsize)
        )

    def forward(self, inputs):
        inputs_ = inputs
        features = self.encoder(inputs_)
        parameters = self.layers(features)

        # print last layers weights
        return features, parameters


class DECA_Encoder(nn.Module):
    def __init__(self, eval_mode=True) -> None:
        super().__init__()

        self.encoder = ResnetEncoder()

        checkpoint = torch.load("assets/deca_model.tar")['E_flame']

        self.encoder.load_state_dict(checkpoint, strict=True)

        self.eval_mode = eval_mode
        if self.eval_mode:
            self.encoder.eval()
        else:
            self.eyelid_layers = nn.Sequential(
                nn.Linear(self.encoder.feature_size, 2),
            )

    def forward(self, img):
        # make B*T
        img_ = img#.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        
        if self.eval_mode:
            with torch.no_grad():
                x = self.encoder(img_)[-1]
            deca_eyelids = None
        else:
            feats, x = self.encoder(img_)
            deca_eyelids = self.eyelid_layers(feats).reshape(img_.size(0), -1)
        # -------- split the parameters -------- #
        deca_shape = x[..., :100]
        deca_tex = x[..., 100:150]
        deca_exp = x[..., 150:200]
        deca_pose = x[..., 200:203]
        deca_jaw = x[..., 203:206]
        deca_cam = x[..., 206:209]
        deca_light = x[..., 209:]

        deca_output = {'shape_params': deca_shape, 'expression_params': deca_exp, 'eyelid_params': deca_eyelids,
                       'jaw_params' : deca_jaw, 'pose_params': deca_pose, 'cam': deca_cam, 'img': img, 
                       'tex': deca_tex, 'light_params': deca_light.reshape(-1, 9, 3) }
        
        return deca_output


class Res_DECA_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = ResnetEncoder()

        checkpoint = torch.load("assets/deca_model.tar")['E_flame']

        self.encoder.load_state_dict(checkpoint, strict=True)
        self.encoder.eval()
        
        self.res_encoder = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=True)
        feature_size = 1280

        self.fnl_layers = nn.Sequential(
            nn.Linear(feature_size, self.encoder.outsize + 2),
        )

        self.fnl_layers[0].weight.data *= 1e-2
        self.fnl_layers[0].bias.data *= 1e-2


    def forward(self, img):
        # make B*T
        img_ = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])

        with torch.no_grad():
            x = self.encoder(img_)[-1]

        res_feats =  self.res_encoder.features(img_)
        res = self.fnl_layers(F.adaptive_max_pool2d(res_feats, (1, 1)).view(img_.size(0), -1))


        # -------- split the parameters -------- #
        deca_shape = x[..., :100] + res[..., :100]
        deca_tex = x[..., 100:150]
        deca_exp = x[..., 150:200] + res[..., 150:200]
        deca_pose = x[..., 200:203] #+ res[..., 200:203]
        deca_jaw = x[..., 203:206] + res[..., 203:206]
        deca_cam = x[..., 206:209] #+ res[..., 206:209]
        deca_light = x[..., 209:]
        deca_eyelids = res[..., -2:]

        deca_output = {'shape_params': deca_shape, 'expression_params': deca_exp, 'eyelid_params': deca_eyelids,
                       'jaw_params' : deca_jaw, 'pose_params': deca_pose, 'cam': deca_cam, 'img': img, 
                       'tex': deca_tex, 'light_params': deca_light.reshape(-1, 9, 3) }
        
        return deca_output