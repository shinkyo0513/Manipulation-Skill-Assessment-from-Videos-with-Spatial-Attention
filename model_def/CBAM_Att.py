import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

# For baseline CBAM Attention (ECCV 2018)

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

        self.rnn_att_size = 128
        self.rnn_top_size = 128

        self.rnn_top = nn.GRUCell(self.x_size, self.rnn_top_size)
        for param in self.rnn_top.parameters():
            if param.dim() > 1:
                torch.nn.init.orthogonal_(param)

        self.x_mid_size = int(feature_size/8)
        self.fc_shrk = nn.Linear(self.x_size, self.x_mid_size, bias=True)
        self.fc_clps = nn.Linear(self.x_mid_size, self.x_size, bias=True)

        self.ins_norm = nn.InstanceNorm2d(2, affine=True)
        self.conv_s1 = nn.Conv2d(2, 32, (3, 3), padding=(1, 1))
        self.conv_s2 = nn.Conv2d(32, 1, (1, 1))

        self.score_fc = nn.Linear(self.rnn_top_size, 1, bias=True)

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
        h_top = torch.randn(batch_size, self.rnn_top_size).to(
            video_tensor.device)
        for frame_idx in range(seq_len):
            # batch_size x 2D x 14 x 14
            featmap = video_tensor[:, frame_idx, :, :, :]

            X = self.relu(self.pre_conv1(featmap))  # batch_size x C x 7 x 7
            X = self.pre_conv2(X)

            x_avg = self.x_avgpool(X).view(batch_size, -1)  # batch_size x C
            x_avg = self.relu(self.fc_shrk(x_avg))
            x_avg = self.fc_clps(x_avg)
            x_max = self.x_maxpool(X).view(batch_size, -1)
            x_max = self.relu(self.fc_shrk(x_max))
            x_max = self.fc_clps(x_max)
            ch_att = self.sigmoid(x_avg+x_max)  # batch_size x D
            ch_att = ch_att.view(batch_size, self.x_size, 1, 1)
            X = X * ch_att  # batch_size x D x 14 x 14

            s_avg = torch.mean(X, dim=1, keepdim=True)
            s_max, _ = torch.max(X, dim=1, keepdim=True)
            # batch_size x 2 x 14 x 14
            s_cat = torch.cat((s_avg, s_max), dim=1)

            s_cat = self.ins_norm(s_cat)
            s_att = self.relu(self.conv_s1(s_cat))
            s_att = self.conv_s2(s_att)  # batch_size x 1 x 7 x 7
            s_att = self.softmax(s_att.view(batch_size, -1)).view(s_att.size())
            video_soft_att.append(s_att.detach().cpu())

            X = X * s_att  # batch_size x C x 7 x 7
            rnn_top_in = torch.sum(
                X.view(batch_size, self.x_size, -1), dim=2)  # batch_size x C
            h_top = self.rnn_top(rnn_top_in, h_top)

        final_score = self.score_fc(h_top).squeeze(1)
        # batch_size x seq_len x 1 x 14 x 14
        video_soft_att = torch.stack(video_soft_att, dim=1)
        return final_score, video_soft_att
