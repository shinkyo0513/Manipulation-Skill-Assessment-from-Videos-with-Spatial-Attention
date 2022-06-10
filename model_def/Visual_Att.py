import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

# For baseline Visual Attention (Arxiv 2015)

class CustomLoss (nn.Module):
    def __init__(self, lamda):
        super(CustomLoss, self).__init__()
        self.lamda = lamda
        self.ranking_loss = nn.MarginRankingLoss(margin=0.5)

    def forward(self, v1_score, v2_score, label, video_soft_att):
        #video_soft_att: batch_size x seq_len x 1 x 7 x 7
        batch_size = video_soft_att.size(0)
        seq_len = video_soft_att.size(1)

        l1 = self.ranking_loss(v1_score, v2_score, label)

        video_soft_att = video_soft_att.view(
            batch_size, seq_len, -1)  # batch_size x seq_len x 49
        l2 = (1-torch.sum(video_soft_att, dim=1))**2  # batch_size x 49
        l2 = torch.mean(l2, dim=1)  # batch_size

        return l1+l2

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
        self.fc_initC = nn.Linear(self.x_size, self.h_size, bias=True)
        self.fc_initH = nn.Linear(self.x_size, self.h_size, bias=True)
        self.rnn_top = nn.LSTMCell(self.x_size, self.h_size)

        self.fc_hl = nn.Linear(self.h_size, 49, bias=False)

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

        video_X = []
        init_vec = 0
        # featmap: batch_size x seq_len x 2D x 14 x 14
        for frame_idx in range(seq_len):
            # batch_size x 2D x 14 x 14
            featmap = video_tensor[:, frame_idx, :, :, :]
            X = self.relu(self.pre_conv1(featmap))  # batch_size x C x 7 x 7
            X = self.relu(self.pre_conv2(X))  # batch_size x C x 7 x 7
            video_X.append(X)
            x_avg = self.x_avgpool(X).view(batch_size, -1)  # batch_size x C
            init_vec += x_avg
        # batch_size x seq_len x C x 7 x 7
        video_X = torch.stack(video_X, dim=1)
        init_vec /= seq_len

        c_top = self.fc_initC(init_vec)  # batch_size x h_size
        h_top = self.fc_initH(init_vec)  # batch_size x h_size

        video_soft_att = []
        for frame_idx in range(seq_len):
            X = video_X[:, frame_idx, :, :, :]  # batch_size x C x 7 x 7

            l = self.fc_hl(h_top)  # batch_size x 49
            l = self.softmax(l)
            s_att = l.view(batch_size, 1, 7, 7)

            X = X * s_att  # batch_size x C x 7 x 7
            video_soft_att.append(s_att.detach().cpu())

            rnn_top_in = torch.sum(
                X.view(batch_size, self.x_size, -1), dim=2)  # batch_size x C
            h_top, c_top = self.rnn_top(rnn_top_in, (h_top, c_top))

        final_score = self.score_fc(h_top).squeeze(1)
        # batch_size x seq_len x 1 x 14 x 14
        video_soft_att = torch.stack(video_soft_att, dim=1)
        return final_score, video_soft_att
