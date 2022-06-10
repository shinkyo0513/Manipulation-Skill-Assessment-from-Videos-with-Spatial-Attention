import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

# For Attention Pooling (NeurIPS 2017)

class model (nn.Module):
    def __init__(self, feature_size, num_seg):
        super(model, self).__init__()
        self.f_size = feature_size
        self.num_seg = num_seg

        self.x_size = 256
        self.pre_conv1 = nn.Conv2d(2*self.f_size, 512, (2, 2), stride=2)
        self.pre_conv2 = nn.Conv2d(512, self.x_size, (1, 1))
        self.x_avgpool = nn.AvgPool2d(7)
        self.x_maxpool = nn.MaxPool2d(7)

        self.bottom_up = nn.Conv2d(self.x_size, 1, (1, 1))
        self.top_down = nn.Conv2d(self.x_size, 1, (1, 1))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(p=0.2)

    # video_featmaps: batch_size x seq_len x D x w x h
    def forward(self, video_tensor):
        batch_size = video_tensor.shape[0]
        seq_len = video_tensor.shape[1]

        video_soft_att = []
        video_score = []
        for frame_idx in range(seq_len):
            # batch_size x 2D x 14 x 14
            featmap = video_tensor[:, frame_idx, :, :, :]

            X = self.relu(self.pre_conv1(featmap))  # batch_size x C x 7 x 7
            X = self.relu(self.pre_conv2(X))  # batch_size x C x 7 x 7

            x_bu = self.bottom_up(X)  # batch_size x 1 x 7 x 7
            x_td = self.top_down(X)  # btahc_size x 1 x 7 x 7

            score = torch.sum(
                (x_bu*x_td).view(batch_size, -1), dim=1)  # batch_size
            video_score.append(score)

            s_att = x_bu * x_td
            video_soft_att.append(s_att.detach().cpu())

        video_score = torch.stack(video_score, dim=1)  # batch_size x seq_len
        final_score = torch.mean(video_score, dim=1)  # batch_size
        # batch_size x seq_len x 1 x 7 x 7
        video_soft_att = torch.stack(video_soft_att, dim=1)
        return final_score, video_soft_att
