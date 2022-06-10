import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

# For baseline SCA-CNN (CVPR 2017)

class model (nn.Module):
    def __init__(self, feature_size, num_seg):
        super(model, self).__init__()
        self.f_size = feature_size
        self.num_seg = num_seg

        self.x_size = 256
        self.pre_conv1 = nn.Conv2d(2*self.f_size, 512, (2, 2), stride=2)
        self.pre_conv2 = nn.Conv2d(512, self.x_size, (1, 1))
        self.x_avgpool = nn.AvgPool2d(7)

        self.h_size = 128
        self.rnn_top = nn.GRUCell(self.x_size, self.h_size)
        for param in self.rnn_top.parameters():
            if param.dim() > 1:
                torch.nn.init.orthogonal_(param)

        self.k_size = int(self.x_size/4)
        self.fc_xc = nn.Linear(1, self.k_size, bias=True)
        self.fc_hc = nn.Linear(self.h_size, self.k_size, bias=False)
        self.fc_b = nn.Linear(self.k_size, 1, bias=True)

        self.conv_s = nn.Conv2d(self.x_size, self.k_size, (1, 1), bias=True)
        self.fc_hs = nn.Linear(self.h_size, self.k_size, bias=False)
        self.fc_a = nn.Linear(self.k_size, 1, bias=True)

        self.score_fc = nn.Linear(self.h_size, 1, bias=True)

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
        h_top = torch.randn(batch_size, self.h_size).to(video_tensor.device)
        for frame_idx in range(seq_len):
            # batch_size x 2D x 14 x 14
            featmap = video_tensor[:, frame_idx, :, :, :]

            X = self.relu(self.pre_conv1(featmap))
            X = self.pre_conv2(X)  # batch_size x C x 7 x 7

            x_avg = self.x_avgpool(X).view(batch_size, -1)  # batch_size x C
            # batch_size x k x C
            tmp_bx = self.fc_xc(x_avg.unsqueeze(-1)).transpose(1, 2)
            tmp_bh = self.fc_hc(h_top).unsqueeze(-1)  # batch_size x k x 1
            b = self.tanh(tmp_bx + tmp_bh)  # batch_size x k x C
            # batch_size x C x 1
            beta = self.sigmoid(self.fc_b(b.transpose(1, 2)))

            ch_att = beta.unsqueeze(-1)  # batch_size x C x 1 x 1
            X = X * ch_att  # batch_size x C x 14 x 14

            tmp_ax = self.conv_s(X).view(
                batch_size, self.k_size, -1)  # batch_size x k x 49
            tmp_ah = self.fc_hs(h_top).unsqueeze(-1)  # batch_size x k x 1
            a = self.tanh(tmp_ax + tmp_ah)  # batch_size x k x 49
            # batch_size x 49 x 1
            alpha = self.softmax(self.fc_a(a.transpose(1, 2)))

            s_att = alpha.view(batch_size, 1, X.size(
                2), X.size(3))  # batch_size x 1 x 7 x 7
            X = X * s_att  # batch_size x C x 7 x 7

            video_soft_att.append(s_att.detach().cpu())
            rnn_top_in = torch.sum(
                X.view(batch_size, self.x_size, -1), dim=2)  # batch_size x C
            h_top = self.rnn_top(rnn_top_in, h_top)

        final_score = self.score_fc(h_top).squeeze(1)
        # batch_size x seq_len x 1 x 14 x 14
        video_soft_att = torch.stack(video_soft_att, dim=1)
        return final_score, video_soft_att
