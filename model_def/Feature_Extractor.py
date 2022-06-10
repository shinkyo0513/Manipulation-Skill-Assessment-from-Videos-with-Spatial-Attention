import os

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

import torchvision
from torchvision.models import resnet101

class feature_extractor (nn.Module):
    def __init__(self, input_type):
        super(feature_extractor, self).__init__()
        assert input_type in ['rgb', 'flow']
        self.input_type = input_type

        self.resnet = resnet101(pretrained=False)
        self.resnet.fc.out_features = 101

        if self.input_type == 'rgb':
            pretrained_wgt = torch.load(os.path.join('model_param', 'ResNet101_rgb_pretrain.pth.tar'))
        elif self.input_type == 'flow':
            pretrained_wgt = torch.load(os.path.join('model_param', 'ResNet101_flow_pretrain.pth.tar'))
        pretrained_wgt = pretrained_wgt['state_dict']
        pretrained_wgt = {k.replace('fc_custom', 'fc'): v for k, v in pretrained_wgt.items()}
        self.resnet.load_state_dict(pretrained_wgt)

        self.feat_extractor = nn.Sequential(*list(self.resnet.children())[:-2]) # Remove avgpool and final fc

    def forward (self, inputs):
        with torch.no_grad():
            bs, ch, nt, h, w = inputs.shape
            frames = torch.cat(torch.unbind(inputs, dim=2), dim=0)  # T*B x C x H x W
            features = self.feat_extractor(frames)  # T*B x 2048 x 14 x 14
            features = torch.stack(features.split(bs, dim=0), dim=1)    # B x T x 2048 x 14 x 14
        return features


