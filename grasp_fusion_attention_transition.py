import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dataset.grasp_dataset import GraspDataset, GraspDataset_Pair
from dataset.grasp_dataset import get_flow_feature_dict, get_rgb_feature_dict

import os
from os.path import join, isdir, isfile, exists
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='full',
                    choices=['full', 'only_x', 'only_htop', 'fc_att', 'no_att', 'cbam', 'sca', 'video_lstm', 'visual'])
parser.add_argument("--feature_type", type=str, default='resnet101_conv5',
                    choices=['resnet101_conv4', 'resnet101_conv5'])
parser.add_argument("--epoch_num", type=int, default=20)
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
#         self.ln = nn.LayerNorm(self.rnn_top_size+self.x_size)
        
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
        h_top = torch.randn(batch_size, self.rnn_top_size).to(video_tensor.device)
        h_att = torch.randn(batch_size, self.rnn_att_size).to(video_tensor.device)
        for frame_idx in range(seq_len):
            featmap = video_tensor[:,frame_idx,:,:,:]  #batch_size x 2D x 14 x 14
            
            X = self.relu(self.pre_conv1(featmap))   #batch_size x C x 7 x 7
            X = self.pre_conv2(X)
            x_avg = self.x_avgpool(X).view(batch_size, -1)   #batch_size x C
            x_max = self.x_maxpool(X).view(batch_size, -1)

            rnn_att_in = torch.cat((self.x_ln(x_avg+x_max),self.h_ln(h_top)), dim=1)
#             rnn_att_in = torch.cat((x_avg+x_max, h_top), dim=1)
#             rnn_att_in = self.ln( torch.cat((x_avg+x_max, h_top), dim=1) )
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

def get_train_test_videos_list (video_name_list, split_index, split_num):
    train_video_list = []
    test_video_list = []

    video_num = len(video_name_list)
    test_video_num = int(math.floor(video_num / split_num))
    test_video_indexs = range(split_index*test_video_num, (split_index+1)*test_video_num)

    for video_index, video_name in enumerate(video_name_list):
        if video_index in test_video_indexs:
            test_video_list.append(video_name)
        else:
            train_video_list.append(video_name)
    return train_video_list, test_video_list

def get_train_test_pairs_dict (dataset_root, train_video_list, test_video_list, cross=True):
    # Read pairs' annotation file
    pairs_annotation_file = open(join(dataset_root, "annotation.txt"), "r")
    all_pairs_dict = {}
    lines = pairs_annotation_file.readlines()
    for line in lines:
        video_name_1, video_name_2, label = line.strip().split(' ')
        all_pairs_dict[tuple((video_name_1, video_name_2))] = int(label)

    train_pairs_dict = {}
    train_videos_num = len(train_video_list)
#     print('training videos num:', train_videos_num)
    for video_index, video_name_1 in enumerate(train_video_list):
        for i in range(video_index+1, train_videos_num):
            video_name_2 = train_video_list[i]
            key = tuple((video_name_1, video_name_2))
            key_inv = tuple((video_name_2, video_name_1))
            if (key in all_pairs_dict) and (all_pairs_dict[key] != 0):
                train_pairs_dict[key] = all_pairs_dict[key]
            elif (key_inv in all_pairs_dict) and (all_pairs_dict[key_inv] != 0):
                train_pairs_dict[key_inv] = all_pairs_dict[key_inv]

    test_pairs_dict = {}
    test_video_num = len(test_video_list)
#     print('validation videos num:', test_video_num)
    for video_index, video_name_1 in enumerate(test_video_list):
        for i in range(video_index+1, test_video_num):
            video_name_2 = test_video_list[i]
            key = tuple((video_name_1, video_name_2))
            key_inv = tuple((video_name_2, video_name_1))
            if (key in all_pairs_dict) and (all_pairs_dict[key] != 0):
                test_pairs_dict[key] = all_pairs_dict[key]
            elif (key_inv in all_pairs_dict) and (all_pairs_dict[key_inv] != 0):
                test_pairs_dict[key_inv] = all_pairs_dict[key_inv]
    if cross:
        for video_name_1 in test_video_list:
            for video_name_2 in train_video_list:
                key = tuple((video_name_1, video_name_2))
                key_inv = tuple((video_name_2, video_name_1))
                if (key in all_pairs_dict) and (all_pairs_dict[key] != 0):
                    test_pairs_dict[key] = all_pairs_dict[key]
                elif (key_inv in all_pairs_dict) and (all_pairs_dict[key_inv] != 0):
                    test_pairs_dict[key_inv] = all_pairs_dict[key_inv]

    return train_pairs_dict, test_pairs_dict

# ============================================================================= #
#                                   main                                        #
# ============================================================================= #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = '../dataset/InfantsGrasping/InfantsGrasping_480x720'
    video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
    video_rgb_feature_dict = get_rgb_feature_dict(dataset_dir, args.feature_type)
    video_flow_feature_dict = get_flow_feature_dict(dataset_dir, args.feature_type)

    num_seg = 25

    split_num = 4
    best_acc_keeper = []
    for split_idx in range(0, split_num):
        print("Split: "+format(split_idx, '01d'))
        train_video_list, test_video_list = get_train_test_videos_list(video_name_list, split_idx, split_num)
        train_pairs_dict, test_pairs_dict = get_train_test_pairs_dict(
                                                            dataset_dir, train_video_list, test_video_list, cross=True)

        dataset_train = GraspDataset_Pair('fusion', video_rgb_feature_dict, video_flow_feature_dict, 
                                                train_pairs_dict, seg_sample=num_seg)
        dataloader_train = DataLoader(dataset_train, batch_size=30, shuffle=True)

        dataset_test = GraspDataset('fusion', video_rgb_feature_dict, video_flow_feature_dict, 
                                          video_name_list, seg_sample=num_seg)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        model_ins = read_model(args.model, args.feature_type, num_seg)

        save_label = f'Grasp/{args.model}/{split_idx:01d}'

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
#                                      lr=5e-6, weight_decay=5e-4, amsgrad=True)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ins.parameters()), 
                                        lr=5e-4, momentum=0.9, weight_decay=1e-2)
    
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        min_loss = 1.0
        no_imprv = 0
        for epoch in range(args.epoch_num):
            train(dataloader_train, model_ins, criterion, optimizer, epoch, device)
            epoch_loss, epoch_acc = test(dataloader_test, test_pairs_dict, model_ins, criterion, epoch, device)
            
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                save_best_result(dataloader_test, test_video_list, model_ins, device, best_acc, save_label)
                
            if epoch_loss <= min_loss:
                min_loss = epoch_loss
                no_imprv = 0
            else:
                no_imprv += 1
            print('Best acc: {:.3f}'.format(best_acc))
            # if no_imprv > 3:
            #     break
        best_acc_keeper.append(best_acc)

    for split_idx, best_acc in enumerate(best_acc_keeper):
        print(f'Split: {split_idx+1}, {best_acc:.4f}')
    print('Avg:', '{:.4f}'.format(sum(best_acc_keeper)/4))
