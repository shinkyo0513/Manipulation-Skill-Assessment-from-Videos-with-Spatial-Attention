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
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='All', choices=['All', 'KnotTying', 'Suturing', 'NeedlePassing'])
parser.add_argument("--feature_type", type=str, default='resnet101_conv5', choices=['resnet101_conv4', 'resnet101_conv5'])

args = parser.parse_args()

def extract_save_rgb_feature (dataset_dir):
    feat_name = args.feature_type.split('_')[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    resnet_rgb_extractor = resnet101(pretrained=True, channel=3, output=feat_name).to(device)
    pretrained_model = torch.load('Models/ResNet101_rgb_pretrain.pth.tar')
    resnet_rgb_extractor.load_state_dict(pretrained_model['state_dict'])
    for param in resnet_rgb_extractor.parameters():
        param.requires_grad = False
    
    if feat_name == 'conv4':
        resize = (224,224)
    elif feat_name == 'conv5':
        resize = (448,448)
    transform = transforms.Compose([
                                transforms.Resize(resize, interpolation=3),
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
        for idx in range(5, seq_len-5, 5):
            batched_frames = video_frames_tensor[:,idx,:,:,:].to(device)   # 1x3x448x448
            with torch.no_grad():
                batched_feature = resnet_rgb_extractor(batched_frames) # 1x2048x14x14
            video_feature.append(batched_feature) 
        video_feature = torch.cat(video_feature, 0).cpu()   # seq_len x 2048 x 14 x 14
        
        feature_save_dir = join(video_dir, 'feature')
        if not os.path.isdir(feature_save_dir):
            os.system('mkdir -p '+feature_save_dir)
        torch.save(video_feature, join(video_dir, 'feature', 'resnet_pretrain_'+feat_name+'.pt'))
        
        videos_feature_dict[video_name] = video_feature
        print(video_feature.size())
        print(video_name+' is finished.')
        utility.save_featmap_heatmaps(video_feature.cpu().data, 
                                      'results/'+args.dataset_name+'_resnet101_'+feat_name+'/', 
                                      (320,240), video_name, dataset_dir)
    return videos_feature_dict

def extract_save_flow_feature (dataset_dir):
    feat_name = args.feature_type.split('_')[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    resnet_flow_extractor = resnet101(pretrained=True, channel=20, output=feat_name).to(device)
    pretrained_model = torch.load('Models/ResNet101_flow_pretrain.pth.tar')
    resnet_flow_extractor.load_state_dict(pretrained_model['state_dict'])
    for param in resnet_flow_extractor.parameters():
        param.requires_grad = False
        
    if feat_name == 'conv4':
        resize = (224,224)
    elif feat_name == 'conv5':
        resize = (448,448)
    transform = transforms.Compose([
                                transforms.Resize(resize, interpolation=3),
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
        for idx in range(0, seq_len-9, 5):
            batched_frames = video_frames_tensor[:,2*idx:2*(idx+10),:,:].to(device)   # 1x20x448x448
            with torch.no_grad():
                batched_feature = resnet_flow_extractor(batched_frames) # 1x2048x14x14
            video_feature.append(batched_feature) 
        video_feature = torch.cat(video_feature, 0).cpu()   # seq_len-9 x 2048 x 14 x 14

        feature_save_dir = join(video_dir, 'feature')
        if not os.path.isdir(feature_save_dir):
            os.system('mkdir -p '+feature_save_dir)
        torch.save(video_feature, join(feature_save_dir, 'resnet_flow_10_pretrain_'+feat_name+'.pt'))

        videos_feature_dict[video_name] = video_feature
        print(video_feature.size())
        print(video_name+' is finished.')
        utility.save_featmap_heatmaps(video_feature.cpu().data, 
                                      'results/'+args.dataset_name+'_resnet101_10_pretrain_'+feat_name+'/', 
                                      (320,240), video_name, dataset_dir)
    return videos_feature_dict

def get_rgb_feature_dict (dataset_dir, feature_type='resnet_conv5'):
    videos_feature_dict = {}
    video_name_list = [video_name for video_name in os.listdir(dataset_dir) 
                                   if isdir(join(dataset_dir, video_name))]
    for video_name in video_name_list:
        video_dir = join(dataset_dir, video_name)
        
        if feature_type == 'resnet101_conv5':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet_pretrain_conv5.pt'))
        elif feature_type == 'resnet101_conv4':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet_pretrain_conv4.pt'))
        
        videos_feature_dict[video_name] = video_feature
    return videos_feature_dict

def get_flow_feature_dict (dataset_dir, feature_type='resnet_conv5'):
    videos_feature_dict = {}
    video_name_list = [video_name for video_name in os.listdir(dataset_dir) 
                                   if isdir(join(dataset_dir, video_name))]
    for video_name in video_name_list:
        video_dir = join(dataset_dir, video_name)
        
        if feature_type == 'resnet101_conv5':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet_flow_10_pretrain_conv5.pt'))
        elif feature_type == 'resnet101_conv4':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet_flow_10_pretrain_conv4.pt'))
        
        videos_feature_dict[video_name] = video_feature
    return videos_feature_dict

# #########################################################################################################################
# #                                                 Surgery_Dataset                                                     #
# #########################################################################################################################
def video_sample (video_tensor, rand_idx_list):
    sampled_video_tensor = torch.stack([video_tensor[idx,:,:,:] for idx in rand_idx_list], dim=0)
    return sampled_video_tensor

class SuguryDataset (Dataset):
    def __init__ (self, ds_root, video_name_list, seg_sample=None):
        self.ds_root = ds_root
        self.video_name_list = video_name_list
        self.seg_sample = seg_sample

        resize = (448, 448)
        self.rgb_transform = transforms.Compose([
            transforms.Resize(resize, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.flow_transform = transforms.Compose([
            transforms.Resize(resize, interpolation=3),
            transforms.ToTensor(),
        ])

    def __len__ (self):
        return len(self.video_name_list)

    def __getitem__ (self, i):
        video_name = self.video_name_list[i]
        video_sample = self.read_one_video(video_name)
        return video_sample

    def read_one_video (self, video_name):
        video_dir = join(self.ds_root, video_name)
        video_rgb_dir = join(video_dir, 'frame')
        video_flow_dir = join(video_dir, 'flow_tvl1')

        seq_len = len(os.listdir(video_rgb_dir))
        rand_idx_list = utility.avg_last_sample(seq_len-10, self.seg_sample)

        rgb_tensors = []
        flow_tensors = []
        sampled_index = []
        for idx in rand_idx_list:
            frame_idx = idx + 5
            rgb_tensors.append(self.rgb_transform(Image.open(
                join(video_rgb_dir, f'{frame_idx:05d}.jpg'))))
            for fi in range(frame_idx-5, frame_idx+5):
                flow_tensors.append(self.flow_transform(Image.open(
                    join(video_flow_dir, f'flow_x_{fi:05d}.jpg'))))
                flow_tensors.append(self.flow_transform(Image.open(
                    join(video_flow_dir, f'flow_y_{fi:05d}.jpg'))))
            sampled_index.append(frame_idx)

        sample = {'name': video_name, 'rgb': rgb_tensors, 'flow': flow_tensors, 'sampled_index': rand_idx_list}
        return sample

class SuguryDataset_Pair (Dataset):
    def __init__ (self, pairs_dict, seg_sample=None):
        self.seg_sample = seg_sample

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
            video_tensor = torch.cat([video_rgb_tensor, video_flow_tensor], dim=1) ##num_seg x 4096 x 14 x 14
              
        rand_idx_list = [5*num+5 for num in rand_idx_list]
        sample = {'name': video_name, 'video': video_tensor, 'sampled_index': rand_idx_list}
        return sample

