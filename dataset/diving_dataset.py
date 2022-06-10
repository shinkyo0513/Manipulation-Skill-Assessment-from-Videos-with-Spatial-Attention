import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import utility
from Models.ResNet import resnet101, resnet34

import os
from os.path import join, isdir, isfile
import glob
import math

from PIL import Image
import pickle
from scipy.io import loadmat
from scipy.stats import spearmanr
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from tqdm import tqdm
import cv2
import csv
import time

def extract_save_rgb_feature (dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    resnet_rgb_extractor = resnet101(pretrained=True, channel=3, output='conv5').to(device)
    pretrained_model = torch.load('Models/ResNet101_rgb_pretrain.pth.tar')
    resnet_rgb_extractor.load_state_dict(pretrained_model['state_dict'])
    for param in resnet_rgb_extractor.parameters():
        param.requires_grad = False
        
    transform = transforms.Compose([
                                transforms.CenterCrop((448,448)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        
    videos_feature_dict = {}
    video_name_list = [format(video_idx, '03d') for video_idx in range(1, 160)]
    for video_name in video_name_list:
        print(video_name+' is being processing.')
        
        video_dir = join(dataset_dir, video_name)
        video_frames_dir = join(video_dir, 'frame')
        video_feature_dir = join(video_dir, 'feature')
        
        if not isdir(video_feature_dir):
            os.system('mkdir '+video_feature_dir)
#         for file in os.listdir(video_feature_dir):
#             if 'heatmaps' not in file:
#                 os.system('rm '+join(video_feature_dir, file))
            
        seq_len = len(os.listdir(video_frames_dir))
        video_frames_tensor = torch.stack([
                                transform(Image.open(join(video_frames_dir, format(frame_idx, '05d')+'.jpg'))) 
                                for frame_idx in range(seq_len)], dim=0)  #1 x seq_len x 3 x 448 x 448
        video_frames_tensor = video_frames_tensor.unsqueeze(0)
        
        # get video's resnet feature maps
        video_feature = []
        for idx in range(seq_len):
            batched_frames = video_frames_tensor[:,idx,:,:,:].to(device)   # 1x3x448x448
            with torch.no_grad():
                batched_feature = resnet_rgb_extractor(batched_frames) # 1x2048x14x14
            video_feature.append(batched_feature) 
        video_feature = torch.cat(video_feature, 0).cpu()   # seq_len x 2048 x 14 x 14
        torch.save(video_feature, join(video_dir, 'feature', 'resnet101_rgb_pretrain_conv5.pt'))
        
        videos_feature_dict[video_name] = video_feature
        print(video_feature.size())
        print(video_name+' is finished.')
        utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/MITDive_resnet101_rgb_pretrain_conv5/', 
                                          (224,224), video_name, dataset_dir)
        
    return videos_feature_dict

def extract_save_flow_feature (dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet_flow_extractor = resnet101(pretrained=True, channel=20, output='conv5').to(device)
    pretrained_model = torch.load('Models/ResNet101_flow_pretrain.pth.tar')
    resnet_flow_extractor.load_state_dict(pretrained_model['state_dict'])
    for param in resnet_flow_extractor.parameters():
        param.requires_grad = False
        
    transform = transforms.Compose([
                                transforms.CenterCrop((448,448)),
                                transforms.ToTensor(),
                                ])
        
    videos_feature_dict = {}
    video_name_list = [format(video_idx, '03d') for video_idx in range(1, 160)]
    for video_name in video_name_list:
        print(video_name+' is being processing.')
        
        video_dir = join(dataset_dir, video_name)
        video_frames_tensor = []
        
        seq_len = int( len(os.listdir(join(video_dir, 'flow_tvl1'))) / 2 )
        for frame_idx in range(1, seq_len+1):
            video_frames_tensor.append(
                transform(Image.open(join(video_dir, 'flow_tvl1', 'flow_x_'+format(frame_idx, '05d')+'.jpg'))))
            video_frames_tensor.append(
                transform(Image.open(join(video_dir, 'flow_tvl1', 'flow_y_'+format(frame_idx, '05d')+'.jpg'))))
        video_frames_tensor = torch.cat(video_frames_tensor, 0).unsqueeze(0)    #1 x seq_lenx2 x 448 x 448
        
        # get video's resnet feature maps
        video_feature = []
        for idx in range(seq_len-9):
            batched_frames = video_frames_tensor[:,2*idx:2*(idx+10),:,:].to(device)   # 1x20x448x448
            with torch.no_grad():
                batched_feature = resnet_flow_extractor(batched_frames) # 1xCxWxH
            video_feature.append(batched_feature) 
        video_feature = torch.cat(video_feature, 0).cpu()   # seq_len-9 x 256 x 28 x 28
        torch.save(video_feature, join(video_dir, 'feature', 'resnet101_flow_10_pretrain_conv5.pt'))
#         os.system('rm '+join(video_dir, 'feature', 'resnet_flow_10_pretrain_conv5.pt'))
        
        videos_feature_dict[video_name] = video_feature
        print(video_feature.size())
        print(video_name+' is finished.')
        utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/MITDive_resnet101_flow_pretrain_conv5/', 
                                          (224,224), video_name, dataset_dir)
    return videos_feature_dict
    
def get_rgb_feature_dict (dataset_dir, feature_type='resnet101_conv5'):
    videos_feature_dict = {}
    video_name_list = [format(video_idx, '03d') for video_idx in range(1, 160)]
    for video_name in video_name_list:
        video_dir = join(dataset_dir, video_name)
        
        if feature_type == 'resnet101_conv5':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet101_rgb_pretrain_conv5.pt'))
        elif feature_type == 'resnet101_conv4':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet101_rgb_pretrain_conv4.pt'))
        
        videos_feature_dict[video_name] = video_feature
    return videos_feature_dict

def get_flow_feature_dict (dataset_dir, feature_type='resnet101_conv5.pt'):
    videos_feature_dict = {}
    video_name_list = [format(video_idx, '03d') for video_idx in range(1, 160)]
    for video_name in video_name_list:
        video_dir = join(dataset_dir, video_name)
        
        if feature_type == 'resnet101_conv5':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet101_flow_10_pretrain_conv5.pt'))
        elif feature_type == 'resnet101_conv4':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet101_flow_10_pretrain_conv4.pt'))
        
        videos_feature_dict[video_name] = video_feature
    return videos_feature_dict

def del_featmaps (dataset_dir):
    video_name_list = [format(video_idx, '03d') for video_idx in range(1, 160)]
    for video_name in video_name_list:
        print(video_name+' is being processing.')
        
        video_dir = join(dataset_dir, video_name)
        video_feature_dir = join(video_dir, 'feature')
    
        for file in os.listdir(video_feature_dir):
            if 'heatmaps' in file or 'resnet101' in file:
                pass
            else:
                print("delete: ", file)
                os.system('rm '+join(video_feature_dir, file))

def del_flow (dataset_dir):
    video_name_list = [format(video_idx, '03d') for video_idx in range(1, 160)]
    for video_name in video_name_list:
        print(video_name+' is being processing.')
        
        video_dir = join(dataset_dir, video_name)
        video_flow_dir = join(video_dir, 'flow')
        
        os.system('rm -rf '+video_flow_dir)
        print('delete: ', video_flow_dir)

#########################################################################################################################
#                                                 MIT_Dive_Dataset                                                      #
#########################################################################################################################
def video_sample (video_tensor, rand_idx_list):
    sampled_video_tensor = torch.stack([video_tensor[idx,:,:,:] for idx in rand_idx_list], dim=0)
    return sampled_video_tensor  

class MITDiveDataset (Dataset):
    def __init__ (self, f_type, video_rgb_feature_dict, video_flow_feature_dict, video_idx_list, seg_sample):
        self.seg_sample = seg_sample
        self.video_rgb_feature_dict = video_rgb_feature_dict
        self.video_flow_feature_dict = video_flow_feature_dict
        self.video_idx_list = video_idx_list
        self.f_type = f_type
        self.overall_scores = np.load('../dataset/MIT_Dive_Dataset/diving_samples_len_ori_800x450/diving_overall_scores.npy')
        self.overall_scores = torch.FloatTensor(np.squeeze(self.overall_scores))

    def __len__ (self):
        return len(self.video_idx_list)

    def __getitem__ (self, i):
        video_idx = self.video_idx_list[i]
        video_name = format(video_idx, '03d')
        video_sample = self.read_one_video(video_name)
        return video_sample

    def read_one_video (self, video_name):
        video_rgb_tensor = self.video_rgb_feature_dict[video_name]
        video_flow_tensor = self.video_flow_feature_dict[video_name]
        
        seq_len = video_flow_tensor.size(0)   #length of optical flow stack (index: 0~N-10)
        
        rand_flow_idx_list = utility.avg_last_sample(seq_len, self.seg_sample)
        video_flow_tensor = video_sample(video_flow_tensor, rand_flow_idx_list) #num_seg x 2048 x 14 x 14

        rand_rgb_idx_list = [flow_idx+5 for flow_idx in rand_flow_idx_list]
        video_rgb_tensor = video_sample(video_rgb_tensor, rand_rgb_idx_list) #num_seg x 2048 x 14 x 14

        if self.f_type == 'flow':
            video_tensor = video_flow_tensor
        elif self.f_type == 'rgb':
            video_tensor = video_rgb_tensor
        elif self.f_type == 'fusion':
            video_tensor = torch.cat([video_rgb_tensor, video_flow_tensor], dim=1) #num_seg x 4096 x 14 x 14
              
        video_score = self.overall_scores[int(video_name)-1]
        sample = {'name': video_name, 'video': video_tensor, 'sampled_index': rand_rgb_idx_list, 'score': video_score}
        return sample
    
#########################################################################################################################
#                                         MIT_Dive_Dataset for Pair                                                     #
#########################################################################################################################   
class MITDiveDataset_Pair(Dataset):
    def __init__(self, f_type, video_rgb_feature_dict, video_flow_feature_dict, pairs_dict, seg_sample):
        self.seg_sample = seg_sample
        self.video_rgb_feature_dict = video_rgb_feature_dict
        self.video_flow_feature_dict = video_flow_feature_dict
        self.f_type = f_type
        
        self.pairs_dict = pairs_dict
        self.pairs_list = list(self.pairs_dict.keys())

    def __len__(self):
        return len(self.pairs_dict)
        
    def __getitem__(self, index):
        v1_idx, v2_idx = self.pairs_list[index]
        v1_name = format(v1_idx, '03d')
        v2_name = format(v2_idx, '03d')
        v1_sample = self.read_one_video(v1_name)
        v2_sample = self.read_one_video(v2_name)
        label = torch.Tensor([ self.pairs_dict[(v1_idx, v2_idx)] ])
        
        pair = {'video1': v1_sample, 'video2': v2_sample, 'label': label}
        return pair

    def read_one_video (self, video_name):
        video_rgb_tensor = self.video_rgb_feature_dict[video_name]
        video_flow_tensor = self.video_flow_feature_dict[video_name]
        
        seq_len = video_flow_tensor.size(0)   #length of optical flow stack (index: 0~N-10)
        
        rand_flow_idx_list = utility.avg_rand_sample(seq_len, self.seg_sample)
        video_flow_tensor = video_sample(video_flow_tensor, rand_flow_idx_list) #num_seg x 2048 x 14 x 14

        rand_rgb_idx_list = [flow_idx+5 for flow_idx in rand_flow_idx_list]
        video_rgb_tensor = video_sample(video_rgb_tensor, rand_rgb_idx_list) #num_seg x 2048 x 14 x 14

        if self.f_type == 'flow':
            video_tensor = video_flow_tensor
        elif self.f_type == 'rgb':
            video_tensor = video_rgb_tensor
        elif self.f_type == 'fusion':
            video_tensor = torch.cat([video_rgb_tensor, video_flow_tensor], dim=1) ##num_seg x 4096 x 14 x 14
              
        sample = {'name': video_name, 'video': video_tensor, 'sampled_index': rand_rgb_idx_list}
        return sample
    
if __name__ == "__main__":
    # extract_save_flow_feature('../dataset/MIT_Dive_Dataset/diving_samples_len_ori_800x450')
    # extract_save_rgb_feature('../dataset/MIT_Dive_Dataset/diving_samples_len_ori_800x450')
    del_flow('../dataset/MIT_Dive_Dataset/diving_samples_len_ori_800x450')