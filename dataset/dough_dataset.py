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
                                transforms.Resize((448,448), interpolation=3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        
    videos_feature_dict = {}
    video_name_list = os.listdir(dataset_dir)
    for video_name in video_name_list:
        print(video_name+' is being processing.')
        
        video_dir = join(dataset_dir, video_name)
        video_frames_dir = join(video_dir, 'frame')
        
        seq_len = len(os.listdir(video_frames_dir))
        video_frames_tensor = torch.stack([
                                transform(Image.open(join(video_frames_dir, format(frame_idx, '05d')+'.jpg'))) 
                                for frame_idx in range(seq_len)], dim=0)  #1 x seq_len x 3 x 448 x 448
        video_frames_tensor = video_frames_tensor.unsqueeze(0)
        
        # get video's resnet feature maps
        video_feature = []
        for idx in range(5, seq_len-5, 50):
            batched_frames = video_frames_tensor[:,idx,:,:,:].to(device)   # 1x3x448x448
            with torch.no_grad():
                batched_feature = resnet_rgb_extractor(batched_frames) # 1x2048x14x14
            video_feature.append(batched_feature) 
        video_feature = torch.cat(video_feature, 0).cpu()   # seq_len x 2048 x 14 x 14
        torch.save(video_feature, join(video_dir, 'feature', 'resnet_pretrain_conv5.pt'))
        
        videos_feature_dict[video_name] = video_feature
        print(video_feature.size())
        print(video_name+' is finished.')
#         utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/chopstick_resnet101_conv5/', 
#                                       (320,180), video_name, dataset_dir)
    return videos_feature_dict

def extract_save_flow_feature (dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    resnet_flow_extractor = resnet101(pretrained=True, channel=20, output='conv5').to(device)
    pretrained_model = torch.load('Models/ResNet101_flow_pretrain.pth.tar')
    resnet_flow_extractor.load_state_dict(pretrained_model['state_dict'])
    for param in resnet_flow_extractor.parameters():
        param.requires_grad = False
        
    transform = transforms.Compose([
                                transforms.Resize((448,448), interpolation=3),
                                transforms.ToTensor(),
                                ])
        
    videos_feature_dict = {}
    video_name_list = os.listdir(dataset_dir)
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
        
        # get video's resnet feature maps, 5-times down-sample
        video_feature = []
        for idx in range(0, seq_len-9, 50):
            batched_frames = video_frames_tensor[:,2*idx:2*(idx+10),:,:].to(device)   # 1x20x448x448
            with torch.no_grad():
                batched_feature = resnet_flow_extractor(batched_frames) # 1x2048x14x14
            video_feature.append(batched_feature) 
        video_feature = torch.cat(video_feature, 0).cpu()   # seq_len-9 x 2048 x 14 x 14
        
        feature_save_dir = join(video_dir, 'feature')
        if not os.path.isdir(feature_save_dir):
            os.system('mkdir -p '+feature_save_dir)
        torch.save(video_feature, join(feature_save_dir, 'resnet_flow_10_pretrain_conv5.pt'))
        
        videos_feature_dict[video_name] = video_feature
        print(video_feature.size())
        print(video_name+' is finished.')
#         utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/chopstick_resnet101_10_pretrain_conv5/', 
#                                       (320,180), video_name, dataset_dir)
    return videos_feature_dict

def get_rgb_feature_dict (dataset_dir, feature_type='resnet101_conv5'):
    videos_feature_dict = {}
    video_name_list = os.listdir(dataset_dir)
    for video_name in video_name_list:
        video_dir = join(dataset_dir, video_name)
        
        if feature_type == 'resnet101_conv5':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet_pretrain_conv5.pt'))
        elif feature_type == 'resnet101_conv4':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet_pretrain_conv4.pt'))
        
        videos_feature_dict[video_name] = video_feature
#         print(video_name, video_feature.size(0))
    return videos_feature_dict

def get_flow_feature_dict (dataset_dir, feature_type='resnet101_conv5'):
    videos_feature_dict = {}
    video_name_list = os.listdir(dataset_dir)
    for video_name in video_name_list:
        video_dir = join(dataset_dir, video_name)
        
        if feature_type == 'resnet101_conv5':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet_flow_10_pretrain_conv5.pt'))
        elif feature_type == 'resnet101_conv4':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet_flow_10_pretrain_conv4.pt'))
        
        videos_feature_dict[video_name] = video_feature
#         print(video_name, video_feature.size(0))
    return videos_feature_dict

def get_train_test_pairs_dict(annotation_dir, split_idx):
    train_pairs_dict = {}
    train_videos = set()
    train_csv = join(annotation_dir, 'DoughRolling_train_' +
                     format(split_idx, '01d')+'.csv')
    with open(train_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row_idx, row in enumerate(csvreader):
            if row_idx != 0:
                key = tuple((row[0], row[1]))
                train_pairs_dict[key] = 1
                train_videos.update(key)
    csvfile.close()

    test_pairs_dict = {}
    test_videos = set()
    test_csv = join(annotation_dir, 'DoughRolling_val_' +
                    format(split_idx, '01d')+'.csv')
    with open(test_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row_idx, row in enumerate(csvreader):
            if row_idx != 0:
                key = tuple((row[0], row[1]))
                test_pairs_dict[key] = 1
                if key[0] not in train_videos:
                    test_videos.add(key[0])
                if key[1] not in train_videos:
                    test_videos.add(key[1])
    csvfile.close()

    return train_pairs_dict, test_pairs_dict, train_videos, test_videos

#########################################################################################################################
#                                                 Chopstick_Dataset                                                     #
#########################################################################################################################
def video_sample (video_tensor, rand_idx_list):
    sampled_video_tensor = torch.stack([video_tensor[idx,:,:,:] for idx in rand_idx_list], dim=0)
    return sampled_video_tensor

class DoughDataset (Dataset):
    def __init__ (self, f_type, video_rgb_feature_dict, video_flow_feature_dict, video_name_list, seg_sample=None):
        self.f_type = f_type
        self.seg_sample = seg_sample
        self.video_rgb_feature_dict = video_rgb_feature_dict
        self.video_flow_feature_dict = video_flow_feature_dict
        self.video_name_list = video_name_list

    def __len__ (self):
        return len(self.video_name_list)

    def __getitem__ (self, i):
        video_name = self.video_name_list[i]
        video_sample = self.read_one_video(video_name)
        return video_sample

    def read_one_video (self, video_name):
        video_rgb_tensor = self.video_rgb_feature_dict[video_name]
        video_flow_tensor = self.video_flow_feature_dict[video_name]
        
        seq_len = video_rgb_tensor.size(0)
        rand_idx_list = utility.avg_last_sample(seq_len, self.seg_sample)

        video_flow_tensor = video_sample(video_flow_tensor, rand_idx_list) #num_seg x 2048 x 14 x 14
        video_rgb_tensor = video_sample(video_rgb_tensor, rand_idx_list) #num_seg x 2048 x 14 x 14
        
        if self.f_type == 'flow':
            video_tensor = video_flow_tensor
        elif self.f_type == 'rgb':
            video_tensor = video_rgb_tensor
        elif self.f_type == 'fusion':
            video_tensor = torch.cat([video_rgb_tensor, video_flow_tensor], dim=1) ##num_seg x 4096 x 14 x 14
          
        rand_idx_list = [50*num+5 for num in rand_idx_list]
        sample = {'name': video_name, 'video': video_tensor, 'sampled_index': rand_idx_list}
        return sample

class DoughDataset_Pair (Dataset):
    def __init__ (self, f_type, video_rgb_feature_dict, video_flow_feature_dict, pairs_dict, seg_sample=None):
        self.f_type = f_type
        self.seg_sample = seg_sample
        self.video_rgb_feature_dict = video_rgb_feature_dict
        self.video_flow_feature_dict = video_flow_feature_dict
        
        self.pairs_dict = pairs_dict
        self.pairs_list = list(self.pairs_dict.keys())

    def __len__ (self):
        return len(self.pairs_list)

    def __getitem__ (self, i):
        v1_name, v2_name = self.pairs_list[i]
        v1_sample = self.read_one_video(v1_name)
        v2_sample = self.read_one_video(v2_name)
        label = torch.Tensor([ self.pairs_dict[(v1_name, v2_name)] ])
        
        pair = {'video1': v1_sample, 'video2': v2_sample, 'label': label}
        return pair

    def read_one_video (self, video_name):
        video_rgb_tensor = self.video_rgb_feature_dict[video_name]
        video_flow_tensor = self.video_flow_feature_dict[video_name]
        
        seq_len = video_rgb_tensor.size(0)
        rand_idx_list = utility.avg_rand_sample(seq_len, self.seg_sample)
        
        video_flow_tensor = video_sample(video_flow_tensor, rand_idx_list) #num_seg x 2048 x 14 x 14
        video_rgb_tensor = video_sample(video_rgb_tensor, rand_idx_list) #num_seg x 2048 x 14 x 14
        
        if self.f_type == 'flow':
            video_tensor = video_flow_tensor
        elif self.f_type == 'rgb':
            video_tensor = video_rgb_tensor
        elif self.f_type == 'fusion':
            video_tensor = torch.cat([video_rgb_tensor, video_flow_tensor], dim=1) #num_seg x 4096 x 14 x 14
              
        rand_idx_list = [50*num+5 for num in rand_idx_list]
        sample = {'name': video_name, 'video': video_tensor, 'sampled_index': rand_idx_list}
        return sample

if __name__ == "__main__":
    extract_save_flow_feature('/data/lzq/dataset/DoughRolling/DoughRolling_600x450')
    extract_save_rgb_feature('/data/lzq/dataset/DoughRolling/DoughRolling_600x450')
