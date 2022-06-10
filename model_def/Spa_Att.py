import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

# For our ICCVW 2019

class model (nn.Module):
    def __init__(self, feature_size, num_seg, variant='full'):
        super(model, self).__init__()
        self.f_size = feature_size
        self.num_seg = num_seg
        self.variant = variant
        assert variant in ['full', 'only_x', 'only_htop', 'fc_att', 'no_att']

        self.x_size = 256
        self.pre_conv1 = nn.Conv2d(2*self.f_size, 512, (2, 2), stride=2)
        self.pre_conv2 = nn.Conv2d(512, self.x_size, (1, 1))
        self.x_avgpool = nn.AvgPool2d(7)
        self.x_maxpool = nn.MaxPool2d(7)

        self.rnn_att_size = 128
        self.rnn_top_size = 128

        if self.variant == 'fc_att':
            self.fc_att = nn.Linear(
                self.x_size+self.rnn_top_size, self.rnn_att_size, bias=True)
        else:
            if self.variant == 'full':
                self.rnn_att = nn.GRUCell(self.x_size+self.rnn_top_size, self.rnn_att_size)
            elif self.variant == 'only_x':
                self.rnn_att = nn.GRUCell(self.x_size, self.rnn_att_size)
            elif self.variant == 'only_htop':
                self.rnn_att = nn.GRUCell(self.rnn_top_size, self.rnn_att_size)
            for param in self.rnn_att.parameters():
                if param.dim() > 1:
                    torch.nn.init.orthogonal_(param)

        self.a_size = 32
        self.xa_fc = nn.Linear(self.x_size, self.a_size, bias=True)
        self.ha_fc = nn.Linear(self.rnn_att_size, self.a_size, bias=True)
        self.a_fc = nn.Linear(self.a_size, 1, bias=False)

        self.rnn_top = nn.GRUCell(self.x_size, self.rnn_top_size)
        for param in self.rnn_top.parameters():
            if param.dim() > 1:
                torch.nn.init.orthogonal_(param)

        self.score_fc = nn.Linear(self.rnn_top_size, 1, bias=True)

        self.x_ln = nn.LayerNorm(self.x_size)
        self.h_ln = nn.LayerNorm(self.rnn_top_size)

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
        h_att = torch.randn(batch_size, self.rnn_att_size).to(
            video_tensor.device)
        for frame_idx in range(seq_len):
            # batch_size x 2D x 14 x 14
            featmap = video_tensor[:, frame_idx, :, :, :]

            X = self.relu(self.pre_conv1(featmap))  # batch_size x C x 7 x 7
            X = self.pre_conv2(X)
            x_avg = self.x_avgpool(X).view(batch_size, -1)  # batch_size x C
            x_max = self.x_maxpool(X).view(batch_size, -1)

            if self.variant == 'no_att':
                rnn_top_in = x_avg
            else:
                if self.variant == 'full':
                    rnn_att_in = torch.cat((self.x_ln(x_avg+x_max), self.h_ln(h_top)), dim=1)
                    # batch_size x rnn_att_size
                    h_att = self.rnn_att(rnn_att_in, h_att)
                elif self.variant == 'only_x':
                    rnn_att_in = self.x_ln(x_avg+x_max)
                    # batch_size x rnn_att_size
                    h_att = self.rnn_att(rnn_att_in, h_att)
                elif self.variant == 'only_htop':
                    rnn_att_in = self.h_ln(h_top)
                    # batch_size x rnn_att_size
                    h_att = self.rnn_att(rnn_att_in, h_att)
                elif self.variant == 'fc_att':
                    rnn_att_in = torch.cat(
                        (self.x_ln(x_avg+x_max), self.h_ln(h_top)), dim=1)
                    h_att = self.fc_att(rnn_att_in)

                # batch_size x 49 x C
                X_tmp = X.view(batch_size, self.x_size, -1).transpose(1, 2)
                # batch_size x 49 x rnn_att_size
                h_att_tmp = h_att.unsqueeze(1).expand(-1, X_tmp.size(1), -1)
                a = self.tanh(self.xa_fc(X_tmp)+self.ha_fc(h_att_tmp))
                a = self.a_fc(a).unsqueeze(2)  # batch_size x 49
                alpha = self.softmax(a)
                s_att = alpha.view(batch_size, 1, X.size(2), X.size(3))
                video_soft_att.append(s_att)

                X = X * s_att  # batch_size x C x 7 x 7
                rnn_top_in = torch.sum(
                    X.view(batch_size, self.x_size, -1), dim=2)  # batch_size x C

            h_top = self.rnn_top(rnn_top_in, h_top)

        final_score = self.score_fc(h_top).squeeze(1)
        if self.variant == 'no_att':
            video_soft_att = torch.zeros(batch_size, seq_len, 1, 14, 14)
        else:
            # batch_size x seq_len x 1 x 14 x 14
            video_soft_att = torch.stack(video_soft_att, dim=1)
        return final_score, video_soft_att
