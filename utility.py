import torch
from torch import nn
from torch.nn import functional as F
import os
from os.path import join, isdir, isfile
import cv2
import numpy as np
import random
import math
import csv

# batched_heatmaps: batch_size x seq_len x 1 x 7 x 7
def save_heatmaps (batched_heatmaps, save_dir, size, video_name, rand_idx_list, t_att, dataset_dir):
    batch_size = batched_heatmaps.size(0)
    seq_len = batched_heatmaps.size(1)

    for batch_offset in range(batch_size):
        att_save_dir = join(save_dir, video_name[batch_offset])
        ori_frames_dir = join(dataset_dir, video_name[batch_offset], 'frame')

        if not os.path.isdir(att_save_dir):
#             os.system('mkdir -p '+att_save_dir)
            os.makedirs(att_save_dir)
        else:
            os.system('rm -rf '+att_save_dir)
#             os.system('mkdir -p '+att_save_dir)
            os.makedirs(att_save_dir)

#         print(rand_idx_list)
        for seq_idx in range(seq_len):
            frame_idx = int(rand_idx_list[seq_idx][batch_offset].item())
            
            heatmap = batched_heatmaps[batch_offset,seq_idx,0,:,:]
            heatmap = (heatmap-heatmap.min()) / (heatmap.max()-heatmap.min())
            heatmap = np.array(heatmap*255.0).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, size)
            
            ori_frame = cv2.imread(join(ori_frames_dir, format(frame_idx, '05d')+'.jpg'))
            if 'diving' in save_dir:
                ori_frame = ori_frame[1:449,176:624]
            ori_frame = cv2.resize(ori_frame, size)
            
            comb = cv2.addWeighted(ori_frame, 0.6, heatmap, 0.4, 0)
#             print(t_att)
            t_att_value = t_att[batch_offset, seq_idx].item()
            pic_save_dir = join(att_save_dir, format(frame_idx, '05d')+'_'+format(t_att_value, '.2f')+'.jpg')
            cv2.imwrite(pic_save_dir, comb)

# featmap: seq_len x 2048 x 14 x 14
def save_featmap_heatmaps (featmap, save_dir, size, video_name, dataset_dir):
    seq_len = featmap.size(0)
    
    s = torch.norm(featmap, p=2, dim=1, keepdim=True)   # seq_len x 1 x 14 x 14
    s = F.normalize(s.view(seq_len, -1),dim=1).view(s.size())
    
    att_save_dir = join(save_dir, video_name)
    ori_frames_dir = join(dataset_dir, video_name, 'frame')
    
    if not os.path.isdir(att_save_dir):
        os.system('mkdir -p '+att_save_dir)
    else:
        os.system('rm -rf '+att_save_dir)
        os.system('mkdir -p '+att_save_dir)
        
    for seq_idx in range(seq_len):
        frame_idx = int(seq_idx)

        heatmap = s[seq_idx,0,:,:]
        heatmap = (heatmap-heatmap.min()) / (heatmap.max()-heatmap.min())
        heatmap = np.array(heatmap*255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, size)

        ori_frame = cv2.imread(join(ori_frames_dir, format(frame_idx, '05d')+'.jpg'))
        if 'Dive' in save_dir:
            ori_frame = ori_frame[1:449,176:624]
        ori_frame = cv2.resize(ori_frame, size)

        comb = cv2.addWeighted(ori_frame, 0.6, heatmap, 0.4, 0)
        pic_save_dir = join(att_save_dir, format(frame_idx, '05d')+'.jpg')
        cv2.imwrite(pic_save_dir, comb)

def save_best_checkpoint (epoch, model, best_acc, save_dir):
    best_checkpoint = {'epoch': epoch,
                       'state_dict': model.state_dict(),
                       'best_acc': best_acc}
    
    if not os.path.isdir(save_dir):
#         os.system('mkdir -p '+save_dir)  
        os.makedirs(save_dir)
    torch.save(best_checkpoint, join(save_dir, 'best_checkpoint.pth.tar'))    

def save_checkpoint (model, epoch, file_dir):
    checkpoint_dir = join(file_dir, format(epoch, '03d'))
    if not os.path.isdir(checkpoint_dir):
        os.system('mkdir -p '+checkpoint_dir)

    torch.save(model.state_dict(), join(checkpoint_dir, 'checkpoint.pth.tar'))
    
def save_record (file_name, type, epoch, epoch_loss, rank_cor):
    new_file = open(file_name, "a") if isfile(file_name) else open(file_name, "w")    
    with new_file:
        writer = csv.writer(new_file)
        writer.writerow(["Epoch: ", epoch, "type: ", type])
        writer.writerow(["epoch_loss: ", format(epoch_loss,".2f"), "rank_cor: ", format(rank_cor,".2f")])

def avg_rand_sample (seq_len, num_seg):
    r = int(seq_len / num_seg)
    real_num_seg = int(math.ceil(seq_len / r))
    
    frame_ind = []
    for i in range(0, real_num_seg-1):
        frame_ind.append(random.randint(i*r, (i+1)*r-1))
    frame_ind.append(random.randint((real_num_seg-1)*r, seq_len-1))
    
    frame_ind = frame_ind[len(frame_ind)-num_seg:]
    return frame_ind

def avg_first_sample (seq_len, num_seg):
    r = int(seq_len / num_seg)
    
    frame_ind = []
    for i in range(0, seq_len, r):
        frame_ind.append(i)
    frame_ind = frame_ind[len(frame_ind)-num_seg:]
    return frame_ind

def avg_last_sample (seq_len, num_seg):
    r = int(seq_len / num_seg)
    
    frame_ind = []
    for i in range(seq_len-1, -1, -r):
        frame_ind.append(i)
    
    frame_ind = frame_ind[0:num_seg]
    frame_ind.reverse()
    return frame_ind

def get_train_test_videos_list (video_record_list, split_index, split_num):
    train_video_list = []
    test_video_list = []

    video_num = len(video_record_list)
    test_video_num = int(math.floor(video_num / split_num))
    test_video_indexs = range(split_index*test_video_num, (split_index+1)*test_video_num)

    for video_index, video_record in enumerate(video_record_list):
        if video_index in test_video_indexs:
            test_video_list.append(video_record)
        else:
            train_video_list.append(video_record)
    return train_video_list, test_video_list

# Ensure the file 'pairs_annotation.txt' exist
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
    for video_index, video_record in enumerate(train_video_list):
        video_name_1 = video_record.video_name
        for i in range(video_index+1, train_videos_num):
            video_name_2 = train_video_list[i].video_name
            key = tuple((video_name_1, video_name_2))
            key_inv = tuple((video_name_2, video_name_1))
            if (key in all_pairs_dict) and (all_pairs_dict[key] != 0):
                train_pairs_dict[key] = all_pairs_dict[key]
            elif (key_inv in all_pairs_dict) and (all_pairs_dict[key_inv] != 0):
                train_pairs_dict[key_inv] = all_pairs_dict[key_inv]

    test_pairs_dict = {}
    test_video_num = len(test_video_list)
#     print('validation videos num:', test_video_num)
    for video_index, video_record in enumerate(test_video_list):
        video_name_1 = video_record.video_name
        for i in range(video_index+1, test_video_num):
            video_name_2 = test_video_list[i].video_name
            key = tuple((video_name_1, video_name_2))
            key_inv = tuple((video_name_2, video_name_1))
            if (key in all_pairs_dict) and (all_pairs_dict[key] != 0):
                test_pairs_dict[key] = all_pairs_dict[key]
            elif (key_inv in all_pairs_dict) and (all_pairs_dict[key_inv] != 0):
                test_pairs_dict[key_inv] = all_pairs_dict[key_inv]
    if cross:
        for video_record_test in test_video_list:
            video_name_1 = video_record_test.video_name
            for video_record_train in train_video_list:
                video_name_2 = video_record_train.video_name
                key = tuple((video_name_1, video_name_2))
                key_inv = tuple((video_name_2, video_name_1))
                if (key in all_pairs_dict) and (all_pairs_dict[key] != 0):
                    test_pairs_dict[key] = all_pairs_dict[key]
                elif (key_inv in all_pairs_dict) and (all_pairs_dict[key_inv] != 0):
                    test_pairs_dict[key_inv] = all_pairs_dict[key_inv]

    return train_pairs_dict, test_pairs_dict

# heatmaps_tensor: seq_len x 1 x w x h
def merge_heatmaps (heatmaps_tensor, num_merge, type='max'):
    if type=='max':
        pooling = nn.MaxPool3d((num_merge,1,1))
    elif type=='avg':
        pooling = nn.AvgPool3d((num_merge,1,1))
        
    seq_len = heatmaps_tensor.size(0)
    heatmaps_tensor = heatmaps_tensor.unsqueeze(0)
    
    heatmaps_tensor = heatmaps_tensor.transpose(1,2)
    merged_heatmaps_tensor = []
    for i in range(0, seq_len-num_merge+1):
        merged_heatmaps_tensor.append(pooling(heatmaps_tensor[:,:,i:i+num_merge,:,:]))
        
    merged_heatmaps_tensor = torch.cat(merged_heatmaps_tensor, 1)
#     print(merged_heatmaps_tensor.shape)
    merged_heatmaps_tensor = merged_heatmaps_tensor.squeeze(0)
    return merged_heatmaps_tensor
    
class HingeL1Loss(nn.Module):
    def __init__ (self, margin=0, size_average=True):
        super(HingeL1Loss, self).__init__()
        self.margin = margin
        self.size_average=True

    def forward (self, input, target):
        d = torch.clamp(torch.abs(input-target)-self.margin, min=0)
        return torch.mean(d) if self.size_average else torch.sum(d)

class SoftAttLoss (nn.Module):
    def __init__ (self, size_average=True):
        super(SoftAttLoss, self).__init__()
        self.size_average = size_average
        
    def forward (self, pred_heatmaps, target_heatmaps):
        batch_size = pred_heatmaps.size(0)
        seq_len = pred_heatmaps.size(1)
        
        pred_heatmaps = F.normalize(pred_heatmaps.view(batch_size, seq_len, -1),dim=2)
        target_heatmaps = F.normalize(target_heatmaps.view(batch_size, seq_len, -1),dim=2)
        l = torch.norm(pred_heatmaps-target_heatmaps, p=2, dim=2)  #batch_size x seq_len
        l = torch.mean(l, dim=1)  #batch_size
        l = torch.mean(l) if self.size_average else torch.sum(l)
        return l
    
class HardAttLoss (nn.Module):
    def __init__ (self, size_average=True):
        super(HardAttLoss, self).__init__()
        self.size_average = size_average
        
    def forward (self, pred_heatmaps):
        batch_size = pred_heatmaps.size(0)
        seq_len = pred_heatmaps.size(1)
        
        hard_att_max, _ = torch.max(pred_heatmaps.view(batch_size, seq_len, -1), dim=2)
        l = 1.0 - hard_att_max  #batch_size x seq_len
        l = torch.mean(l, dim=1)
        l = torch.mean(l) if self.size_average else torch.sum(l)
        return l

class OuterAttLoss (nn.Module):
    def __init__ (self, size_average=True):
        super(OuterAttLoss, self).__init__()
        self.size_average = size_average
        
    def forward (self, pred_heatmaps, target_heatmaps):
        batch_size = pred_heatmaps.size(0)
        seq_len = pred_heatmaps.size(1)
        
        
#         pred_heatmaps = F.normalize(pred_heatmaps.view(batch_size, seq_len, -1),dim=2)
        pred_heatmaps = pred_heatmaps.view(batch_size, seq_len, -1)
#         target_heatmaps = F.normalize(target_heatmaps.view(batch_size, seq_len, -1),dim=2)
        target_heatmaps = target_heatmaps.view(batch_size, seq_len, -1)
        
        outer = (target_heatmaps==0).to(dtype=torch.float)
        outer = pred_heatmaps*outer #batch_size x seq_len x 14*14
        l = torch.sum(outer, dim=2)  #batch_size x seq_len
#         l = torch.norm(outer, p=2, dim=2)
        l = torch.mean(l, dim=1)  #batch_size
        l = torch.mean(l) if self.size_average else torch.sum(l)
        return l
# ============================================================================= #
#                       Video info store class                                  #
# ============================================================================= #   
class VideoRecord(object):
    def __init__(self, file_name):
        self._file_name = file_name
        self._data = file_name.strip().split('_')
        
    @property
    def label(self):
        return float(self._data[2][1:])

    @property
    def frame_rate(self):
        return int(self._data[5])

    @property
    def video_name(self):
        return str(self._file_name)
        
    @property
    def video_len(self):
        return int(self._data[4])-int(self._data[3]) + 1 