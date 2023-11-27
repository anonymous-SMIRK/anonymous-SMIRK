import torch
from .resnet import resnet50, load_state_dict
import torch.nn as nn
import torch.nn.functional as F

class VGGFace2Loss(nn.Module):
    def __init__(self):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval().cuda()
        load_state_dict(self.reg_model, 'assets/resnet50_ft_weight.pkl')
        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).cuda()

    def reg_features(self, x):
        # out = []
        margin=10
        x = x[:,:,margin:224-margin,margin:224-margin]
        # x = F.interpolate(x*2. - 1., [224,224], mode='nearest')
        x = F.interpolate(x*2. - 1., [224,224], mode='bilinear')
        # import ipdb; ipdb.set_trace()
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # import ipdb;ipdb.set_trace()
        img = img[:, [2,1,0], :, :].permute(0,2,3,1) * 255 - self.mean_bgr
        img = img.permute(0,3,1,2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True, use_mean=True, metric='cos'):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        # loss = ((gen_out - tar_out)**2).mean()
        
        if metric == 'cos':
            loss = self._cos_metric(gen_out, tar_out)
        elif metric == 'l1':
            loss = torch.abs(gen_out - tar_out).mean(dim=1)
        elif metric == 'l2':
            loss = ((gen_out - tar_out)**2).mean(dim=1)
        else:
            raise NotImplementedError

        if use_mean:
            loss = loss.mean()
            
        return loss