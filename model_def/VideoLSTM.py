import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

# For baseline Video LSTM 

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class model (nn.Module):
    def __init__(self, feature_size, num_seg):
        super(model, self).__init__()
        self.f_size = feature_size
        self.num_seg = num_seg
        self.f_width = 14
        self.f_height = 14

        self.x_size = 512
        self.conv_rgb = nn.Conv2d(self.f_size, self.x_size, (2, 2), stride=2)
        self.conv_flow = nn.Conv2d(self.f_size, self.x_size, (2, 2), stride=2)

        self.htop_size = 128
        self.hatt_size = 128

        self.top_lstm = ConvLSTMCell((7, 7), self.x_size, self.htop_size,
                                     kernel_size=(3, 3), bias=True)
        self.att_lstm = ConvLSTMCell((7, 7), self.x_size+self.htop_size, self.hatt_size,
                                     kernel_size=(3, 3), bias=True)

        self.conv_z1 = nn.Conv2d(
            self.x_size+self.hatt_size, 256, (1, 1), bias=True)
        self.conv_z2 = nn.Conv2d(256, 1, (1, 1), bias=False)

        self.avgpool = nn.AvgPool2d(7)
        self.score_fc = nn.Linear(self.htop_size*49, 1, bias=True)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(p=0.3)

    # video_featmaps: batch_size x seq_len x D x w x h
    def forward(self, video_tensor):
        batch_size = video_tensor.shape[0]
        seq_len = video_tensor.shape[1]

        video_soft_att = []
#         video_htop = []
        htop = torch.randn(batch_size, self.htop_size,
                           7, 7).to(video_tensor.device)
        ctop = torch.randn(batch_size, self.htop_size,
                           7, 7).to(video_tensor.device)
        hatt = torch.randn(batch_size, self.hatt_size,
                           7, 7).to(video_tensor.device)
        catt = torch.randn(batch_size, self.hatt_size,
                           7, 7).to(video_tensor.device)
        for frame_idx in range(seq_len):
            # batch_size x 2D x 14 x 14
            featmap = video_tensor[:, frame_idx, :, :, :]
#             rgb, flow = torch.split(featmap, self.f_size, dim=1)
            rgb = self.conv_rgb(featmap[:, :2048, :, :])
            flow = self.conv_flow(featmap[:, 2048:, :, :])

            att_lstm_in = torch.cat((flow, htop), dim=1)
            hatt, catt = self.att_lstm(att_lstm_in, (hatt, catt))

            z = self.conv_z1(torch.cat((flow, hatt), dim=1))
            z = self.tanh(z)
            z = self.conv_z2(z)
            s_att = self.softmax(z.view(batch_size, -1)
                                 ).view(batch_size, 1, 7, 7)
            video_soft_att.append(s_att)

            top_lstm_in = rgb * s_att
            htop, ctop = self.top_lstm(top_lstm_in, (htop, ctop))
#             video_htop.append(htop)

#         video_htop = torch.stack(video_htop, dim=1)
#         tmp = torch.mean(video_htop, dim=1, keepdim=False).view(batch_size, -1)
        tmp = htop.view(batch_size, -1)
        tmp = self.dropout(tmp)
        final_score = self.score_fc(tmp).squeeze(1)
        # batch_size x seq_len x 1 x 14 x 14
        video_soft_att = torch.stack(video_soft_att, dim=1)
        return final_score, video_soft_att