from logging import raiseExceptions
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset.drawing_dataset import DrawingDataset, DrawingDataset_Pair
from dataset.drawing_dataset import get_flow_feature_dict, get_rgb_feature_dict
from common import train, test, save_best_result

import os
from os.path import join, isdir, isfile, exists
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='All',
                    choices=['All', 'HandDrawing', 'SonicDrawing'])
parser.add_argument("--model", type=str, default='full',
                    choices=['full', 'only_x', 'only_htop', 'fc_att', 'no_att', 'cbam', 'sca', 'video_lstm', 'visual'])
parser.add_argument("--feature_type", type=str, default='resnet101_conv5', choices=['resnet101_conv4', 'resnet101_conv5'])
parser.add_argument("--epoch_num", type=int, default=30, choices=[10,20,30])
parser.add_argument("--split_index", type=int, default=0, choices=[0,1,2,3,4])
parser.add_argument("--label", type=str, default='')

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
        
        self.x_ln = nn.LayerNorm(self.x_size)
        self.h_ln = nn.LayerNorm(self.rnn_top_size)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(p=0.1)
        
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

            rnn_att_in = torch.cat((self.x_ln(x_avg+x_max),self.h_ln(h_top)), dim=1)
#             rnn_att_in = torch.cat((x_avg+x_max, h_top), dim=1)
            h_att = self.rnn_att(rnn_att_in, h_att)    #batch_size x rnn_att_size
            
            X_tmp = X.view(batch_size, self.x_size, -1).transpose(1,2)   #batch_size x 49 x C
            h_att_tmp = h_att.unsqueeze(1).expand(-1,X_tmp.size(1),-1)    #batch_size x 49 x rnn_att_size
            
            a = self.tanh(self.xa_fc(X_tmp)+self.ha_fc(h_att_tmp))
            a = self.a_fc(a).unsqueeze(2)   #batch_size x 49
            alpha = self.softmax(a)
            s_att = alpha.view(batch_size, 1, X.size(2), X.size(3))
            video_soft_att.append(s_att)
            
            X = X * s_att   #batch_size x C x 7 x 7
            rnn_top_in = torch.sum(X.view(batch_size, self.x_size, -1), dim=2)   #batch_size x C x 7 x 7
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

def get_train_test_pairs_dict (annotation_dir, dataset_name, split_idx):
    train_pairs_dict = {}
    train_videos = set()
    train_csv = join(annotation_dir, dataset_name+'_train_'+format(split_idx, '01d')+'.csv')
    with open(train_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row_idx, row in enumerate(csvreader):
            if row_idx != 0:
                key = tuple((dataset_name+'_'+row[0], dataset_name+'_'+row[1]))
                train_pairs_dict[key] = 1
                train_videos.update(key)
    csvfile.close()
    
    test_pairs_dict = {}
    tets_videos = set()
    test_csv = join(annotation_dir, dataset_name+'_val_'+format(split_idx, '01d')+'.csv')
    with open(test_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row_idx, row in enumerate(csvreader):
            if row_idx != 0:
                v0 = dataset_name+'_'+row[0]
                v1 = dataset_name+'_'+row[1]
                key = tuple((v0, v1))
                test_pairs_dict[key] = 1
                if v0 not in train_videos:
                    test_videos.add(v0)
                if v1 not in train_videos:
                    test_videos.add(v1)
    csvfile.close()
                
    return train_pairs_dict, test_pairs_dict, train_videos, test_videos

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_name_list = ['SonicDrawing', 'HandDrawing'] if args.dataset_name=='All' else [args.dataset_name]
    
    # read name_list & feature dict of all the videos
    video_name_list = []
    video_rgb_feature_dict = {}
    video_flow_feature_dict = {}
    for dataset_name in dataset_name_list:
        dataset_dir = join('../dataset', dataset_name, dataset_name+'_Stationary_800x450')
        sub_video_name_list = [video_name for video_name in os.listdir(dataset_dir) if isdir(join(dataset_dir, video_name))]
        sub_video_rgb_feature_dict = get_rgb_feature_dict(dataset_dir, args.feature_type)
        sub_video_flow_feature_dict = get_flow_feature_dict(dataset_dir, args.feature_type)
        for video_name in sub_video_name_list:
            sub_video_rgb_feature_dict[dataset_name+'_'+video_name] = sub_video_rgb_feature_dict.pop(video_name)
            sub_video_flow_feature_dict[dataset_name+'_'+video_name] = sub_video_flow_feature_dict.pop(video_name)
        sub_video_name_list = [dataset_name+'_'+video_name for video_name in sub_video_name_list]
            
        video_name_list += sub_video_name_list
        video_rgb_feature_dict.update(sub_video_rgb_feature_dict)
        video_flow_feature_dict.update(sub_video_flow_feature_dict)
    del sub_video_name_list, sub_video_rgb_feature_dict, sub_video_flow_feature_dict
    
    best_acc_keeper = []
    for split_idx in range(1, 5):
        print("Split: "+format(split_idx, '01d'))
        
        # read pairs dict of videos belonging to this split
        train_pairs_dict = {}
        test_pairs_dict = {}
        train_videos = set()
        test_videos = set()
        for dataset_name in dataset_name_list:
            annotation_dir = join('../dataset', dataset_name, dataset_name+'_Annotation/splits')
            sub_train_pairs_dict, sub_test_pairs_dict, sub_train_videos, sub_test_videos = get_train_test_pairs_dict(
                annotation_dir, dataset_name, split_idx)
            
            train_pairs_dict.update(sub_train_pairs_dict)
            test_pairs_dict.update(sub_test_pairs_dict)
            train_videos.update(sub_train_videos)
            test_videos.update(sub_test_videos)
        del sub_train_pairs_dict, sub_test_pairs_dict, sub_train_videos, sub_test_videos
            
        num_seg = 25
        dataset_train = DrawingDataset_Pair('fusion', video_rgb_feature_dict, video_flow_feature_dict, 
                                                  train_pairs_dict, seg_sample=num_seg)
        dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

        dataset_test = DrawingDataset('fusion', video_rgb_feature_dict, video_flow_feature_dict, 
                                              video_name_list, seg_sample=num_seg)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        model_ins = read_model(args.model, args.feature_type, num_seg)

        save_label = f'Drawing_{args.dataset_name}/{args.model}/{split_idx:01d}'

        best_acc = 0.0
        if args.continue_train:
            ckpt_dir = join('checkpoints', save_label,
                            'best_checkpoint.pth.tar')
            if exists(checkpoint):
                checkpoint = torch.load(ckpt_dir)
                model_ins.load_state_dict(checkpoint['state_dict'])
                best_acc = checkpoint['best_acc']
                print("Start from previous checkpoint, with rank_cor: {:.4f}".format(
                    checkpoint['best_acc']))
            else:
                print("No previous checkpoint. \nStart from scratch.")
        else:
            print("Start from scratch.")

        model_ins.to(device)

        criterion = nn.MarginRankingLoss(margin=0.5)

#         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_ins.parameters()), 
#                                      lr=1e-5, weight_decay=5e-4, amsgrad=True)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ins.parameters()), 
                                            lr=1e-3, momentum=0.9, weight_decay=1e-3)    #real l2 reg = weight_decay*lr

#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        min_loss = 1.0
        no_imprv = 0
        for epoch in range(args.epoch_num):
            train(dataloader_train, model_ins, criterion, optimizer, epoch, device)
            epoch_loss, epoch_acc = test(dataloader_test, test_pairs_dict, model_ins, criterion, epoch, device)
            
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                save_best_result(dataloader_test, test_videos,
                                 model_ins, device, best_acc, save_label)
                
            if epoch_loss <= min_loss:
                min_loss = epoch_loss
                no_imprv = 0
            else:
                no_imprv += 1
            print('Best acc: {:.3f}'.format(best_acc))
#             if no_imprv > 3:
#                 break
        best_acc_keeper.append(best_acc)
        
    for split_idx, best_acc in enumerate(best_acc_keeper):
        print(f'Split: {split_idx+1}, {best_acc:.4f}')
    print('Avg:', '{:.4f}'.format(sum(best_acc_keeper)/4))
