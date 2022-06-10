import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset.diving_dataset import MITDiveDataset, MITDiveDataset_Pair
from dataset.diving_dataset import get_flow_feature_dict, get_rgb_feature_dict
import utility

import os
from os.path import join, isdir, isfile
import glob
import math
import argparse

from PIL import Image
import pickle
from scipy.io import loadmat
from scipy.stats import spearmanr
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from tqdm import tqdm
import cv2
import csv
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='full',
                    choices=['full', 'only_x', 'only_htop', 'fc_att', 'no_att', 'cbam', 'sca', 'video_lstm', 'visual'])
parser.add_argument("--feature_type", type=str, default='resnet101_conv5', choices=['resnet101_conv4', 'resnet101_conv5'])
parser.add_argument("--epoch_num", type=int, default=30)
parser.add_argument("--split_index", type=int, default=0, choices=[0,1,2,3,4])
parser.add_argument("--label", type=str, default='Full_model')

args = parser.parse_args()

'''
class model (nn.Module):
    def __init__ (self, feature_size, num_seg):
        super(model, self).__init__()
        self.f_size = feature_size
        self.num_seg = num_seg
        
        self.x_size = 256
        self.pre_conv1 = nn.Conv2d(2*self.f_size, 512, (2,2), stride=2)
        self.pre_conv2 = nn.Conv2d(512, self.x_size, (1,1))
        self.x_avgpool = nn.AvgPool2d(7)
        self.x_maxpool = nn.MaxPool2d(7)
        
        self.rnn_att_size = 128
        self.rnn_top_size = 128
        
        self.rnn_top = nn.GRUCell(self.x_size, self.rnn_top_size)
        for param in self.rnn_top.parameters():
            if param.dim() > 1:
                torch.nn.init.orthogonal_(param)
        
        self.rnn_att = nn.GRUCell(self.x_size+self.rnn_top_size, self.rnn_att_size)
        for param in self.rnn_att.parameters():
            if param.dim() > 1:
                torch.nn.init.orthogonal_(param)
                
        self.a_size = 32
        self.xa_fc = nn.Linear(self.x_size, self.a_size, bias=True)
        self.ha_fc = nn.Linear(self.rnn_att_size, self.a_size, bias=True)
        self.a_fc = nn.Linear(self.a_size, 1, bias=False)
        
        self.score_fc = nn.Linear(self.rnn_top_size, 1, bias=True)
        
#         self.x_ln = nn.LayerNorm(self.x_size)
#         self.h_ln = nn.LayerNorm(self.rnn_top_size)
        self.ln = nn.LayerNorm(self.rnn_top_size+self.x_size)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(p=0.2)
        
    # video_featmaps: batch_size x seq_len x D x w x h
    def forward (self, video_tensor):
        batch_size = video_tensor.shape[0]
        seq_len = video_tensor.shape[1]
        
        video_soft_att = []
        h_top = torch.zeros(batch_size, self.rnn_top_size).to(video_tensor.device)
        h_att = torch.zeros(batch_size, self.rnn_att_size).to(video_tensor.device)
        for frame_idx in range(seq_len):
            featmap = video_tensor[:,frame_idx,:,:,:]  #batch_size x 2D x 14 x 14
            
            X = self.relu(self.pre_conv1(featmap))   #batch_size x C x 7 x 7
            X = self.pre_conv2(X)
            x_avg = self.x_avgpool(X).view(batch_size, -1)   #batch_size x C
            x_max = self.x_maxpool(X).view(batch_size, -1)

#             rnn_att_in = torch.cat((self.x_ln(x_avg+x_max),self.h_ln(h_top)), dim=1)
#             rnn_att_in = torch.cat((x_avg+x_max, h_top), dim=1)
            rnn_att_in = self.ln( torch.cat((x_avg+x_max, h_top), dim=1) )
            h_att = self.rnn_att(rnn_att_in, h_att)    #batch_size x rnn_att_size
            
            X_tmp = X.view(batch_size, self.x_size, -1).transpose(1,2)   #batch_size x 49 x C
            h_att_tmp = h_att.unsqueeze(1).expand(-1,X_tmp.size(1),-1)    #batch_size x 49 x rnn_att_size
            a = self.tanh(self.xa_fc(X_tmp)+self.ha_fc(h_att_tmp))
            a = self.a_fc(a).unsqueeze(2)   #batch_size x 49
            alpha = self.softmax(a)
            s_att = alpha.view(batch_size, 1, X.size(2), X.size(3))
            video_soft_att.append(s_att)
            
            X = X * s_att   #batch_size x C x 7 x 7
            rnn_top_in = torch.sum(X.view(batch_size, self.x_size, -1), dim=2)   #batch_size x C
            h_top = self.rnn_top(rnn_top_in, h_top)
            
        final_score = self.score_fc(h_top).squeeze(1)
        video_soft_att = torch.stack(video_soft_att, dim=1)  #batch_size x seq_len x 1 x 14 x 14
        video_tmpr_att = torch.zeros(batch_size, seq_len)
        return final_score, video_soft_att, video_tmpr_att
'''


def read_model(model_type, feature_type, num_seg):
    feature_size = 2048 if feature_type == 'resnet101_conv5' else 1024
    if model_type in ['full', 'only_x', 'only_htop', 'fc_att', 'no_att']:
        from model_def.Spa_Att import model
        return model(feature_size, num_seg, variant=model_type)
    elif model_type in ['cbam']:
        from model_def.CBAM_Att import model
        return model(feature_size, num_seg)
    elif model_type in ['sca']:
        from model_def.SCA_Att import model
        return model(feature_size, num_seg)
    elif model_type in ['video_lstm']:
        from model_def.VideoLSTM import model
        return model(feature_size, num_seg)
    elif model_type in ['visual']:
        from model_def.Visual_Att import model
        return model(feature_size, num_seg)
    else:
        raise Exception(f'Unsupport model type of {model_type}.')

##################################################
#                  Train & Test                  #
##################################################
def train (dataloader, model, criterion, optimizer, epoch, device, write_txt=False):
    model.train()
    
    running_loss = 0.0
    running_acc = 0.0

    for batch_idx, pair in enumerate(tqdm(dataloader)):
        v1_tensor = pair['video1']['video'].to(device)
        v2_tensor = pair['video2']['video'].to(device)
        label = pair['label'].to(device).squeeze(1)

        v1_score, v1_satt, v1_tatt = model(v1_tensor)
        v2_score, v2_satt, v2_tatt = model(v2_tensor)
        
        loss = criterion(v1_score, v2_score, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()   
        running_acc += torch.nonzero((label*(v1_score-v2_score))>0).size(0) / v1_tensor.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    output = 'Epoch:{} '.format(epoch,)+'Train, Loss:{:.4f}, Acc:{:.4f}'.format(epoch_loss, epoch_acc)
    print(output)
    
    if write_txt:
        file = open("results/diving_fusion_attention_transition.txt", "a")
        if epoch==0:
            file.write("============================================================================\n")
        file.write(output+'\n')
        file.close()

def test (dataloader, pairs_dict, model, criterion, epoch, device, write_txt):
    model.eval()   
    
    pred_score_list = torch.Tensor([]).to(dtype=torch.float32)
    gt_score_list = torch.Tensor([]).to(dtype=torch.float32)
    
    videos_score = {}
    for batch_idx, sample in enumerate(tqdm(dataloader)):
        video_name = sample['name']
        gt_score = sample['score']
        v_tensor = sample['video'].to(device)
        sampled_idx_list = sample['sampled_index']
        
        with torch.no_grad():
            v_score, v_satt, v_tatt = model(v_tensor)
        for i in range(v_tensor.size(0)):
            videos_score[video_name[i]] = v_score[i].unsqueeze(0)
            
        pred_score_list = torch.cat((pred_score_list, v_score.cpu().data), 0)
        gt_score_list = torch.cat((gt_score_list, gt_score.cpu().data), 0)
    
    running_loss = 0.0
    running_acc = 0.0
    
    pairs_list = list(pairs_dict.keys())
    for v1_idx, v2_idx in pairs_list:
        v1_name = format(v1_idx, '03d')
        v2_name = format(v2_idx, '03d')
        v1_score = videos_score[v1_name]
        v2_score = videos_score[v2_name]
        label = torch.Tensor([ pairs_dict[(v1_idx, v2_idx)] ]).to(device)
        
        loss = criterion(v1_score, v2_score, label)
        
        running_loss += loss.item()
        running_acc += torch.nonzero((label*(v1_score-v2_score))>0).size(0)
    
    epoch_loss = running_loss / len(pairs_list)
    epoch_acc = running_acc / len(pairs_list)
    
    rankcorr, _ = spearmanr(pred_score_list, gt_score_list)
    
    output = 'Epoch:{} '.format(epoch,)+'Test, Loss:{:.4f}, Acc:{:.4f}, RankCor:{:.4f}'.format(epoch_loss, epoch_acc, rankcorr)
    print(output)
    
    if write_txt:
        file = open("results/diving_fusion_attention_transition.txt", "a")
        file.write(output+'\n')
        file.close()
        
#     print(pred_score_list)  
    return epoch_loss, epoch_acc, rankcorr

def save_best_result (dataloader, model, device, best_rankcorr, dataset_dir, test_video_idx_list):
    model.eval()
    file_name = 'diving_tr/'+args.label
    
    utility.save_best_checkpoint(epoch, model, best_rankcorr, join('checkpoints',file_name))

    videos_score = {}
    for sample in dataloader:
        video_name = sample['name']
        v_tensor = sample['video'].to(device)
        gt_score = sample['score']
        sampled_idx_list = sample['sampled_index']

        with torch.no_grad():
            v_score, v_satt, v_tatt = model(v_tensor)
        for i in range(v_tensor.size(0)):
            videos_score[video_name[i]] = [v_score[i].item(), gt_score[i].item()]

        utility.save_heatmaps(v_satt.cpu().data, join('results',file_name), (320,320), 
                                  video_name, sampled_idx_list, v_tatt.cpu().data, dataset_dir)

    os.system('mkdir -p '+join('results/videos_score',file_name))
    with open(join('results/videos_score',file_name,'0.pickle'),'wb') as f:
        pickle.dump(videos_score, f)
    f.close() 

def get_train_test_pairs_dict (dataset_root, train_video_idx_list, test_video_idx_list, cross=False):
    overall_scores = np.load(join(dataset_root, 'diving_overall_scores.npy'))
    overall_scores = torch.FloatTensor(np.squeeze(overall_scores))
    
    score_diff_thres = 5.0
    
    train_pairs_dict = {}
    for idx_1, video_idx_1 in enumerate(train_video_idx_list):
        for idx_2 in range(idx_1+1, len(train_video_idx_list)):
            video_idx_2 = train_video_idx_list[idx_2]
            key = tuple((video_idx_1, video_idx_2))
            score_diff = overall_scores[video_idx_1-1] - overall_scores[video_idx_2-1]
            if score_diff > score_diff_thres:
                label = 1.0
                train_pairs_dict[key] = label
            elif score_diff < -score_diff_thres:
                label = -1.0
                train_pairs_dict[key] = label

    test_pairs_dict = {}
    for idx_1, video_idx_1 in enumerate(test_video_idx_list):
        for idx_2 in range(idx_1+1, len(test_video_idx_list)):
            video_idx_2 = test_video_idx_list[idx_2]
            key = tuple((video_idx_1, video_idx_2))
            score_diff = overall_scores[video_idx_1-1] - overall_scores[video_idx_2-1]
            if score_diff > score_diff_thres:
                label = 1.0
                test_pairs_dict[key] = label
            elif score_diff < -score_diff_thres:
                label = -1.0
                test_pairs_dict[key] = label
    if cross:
        for video_idx_1 in test_video_idx_list:
            for video_idx_2 in train_video_idx_list:
                key = tuple((video_idx_1, video_idx_2))
                score_diff = overall_scores[video_idx_1-1] - overall_scores[video_idx_2-1]
                if score_diff > score_diff_thres:
                    label = 1.0
                    test_pairs_dict[key] = label
                elif score_diff < -score_diff_thres:
                    label = -1.0
                    test_pairs_dict[key] = label

    return train_pairs_dict, test_pairs_dict

# ============================================================================= #
#                                   main                                        #
# ============================================================================= #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = '../dataset/MIT_Dive_Dataset/diving_samples_len_ori_800x450'
    video_rgb_feature_dict = get_rgb_feature_dict(dataset_dir, 'resnet101_conv5')
    video_flow_feature_dict = get_flow_feature_dict(dataset_dir, 'resnet101_conv5')
    
    video_idx_list = list(range(1, 160))    
    train_video_idx_list = list(range(1, 101))
    test_video_idx_list = list(range(101, 160))
    train_pairs_dict, test_pairs_dict = get_train_test_pairs_dict(
                                            dataset_dir, train_video_idx_list, test_video_idx_list, cross=False)

    num_seg = 25
    dataset_train = MITDiveDataset_Pair('fusion', video_rgb_feature_dict, video_flow_feature_dict, 
                                            train_pairs_dict, num_seg)
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

    dataset_test = MITDiveDataset('fusion', video_rgb_feature_dict, video_flow_feature_dict, 
                                      test_video_idx_list, num_seg)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # model_ins = model(2048, num_seg)
    # model_ins.to(device)
    model_ins = read_model(args.model, args.feature_type, num_seg)

    criterion = nn.MarginRankingLoss(margin=0.5)

#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_ins.parameters()), 
#                                      lr=1e-6, weight_decay=5e-4, amsgrad=True)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ins.parameters()), 
                                    lr=5e-5, momentum=0.9, weight_decay=1e-3)
    
    best_rankcorr = -1.1
    for epoch in range(20):
        train(dataloader_train, model_ins, criterion, optimizer, epoch, device, write_txt=True)
        epoch_loss, epoch_acc, rankcorr = test(dataloader_test, test_pairs_dict, 
                                               model_ins, criterion, epoch, device, write_txt=True)
        if rankcorr > best_rankcorr:
            best_rankcorr = rankcorr
            save_best_result(dataloader_test, model_ins, device, best_rankcorr, dataset_dir, test_video_idx_list)
        print('best rankcorr: {:.3f}'.format(best_rankcorr))
