import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import utility
from Models.ResNet import resnet101
# from Models.c3d_model import C3D
# from Models import resnet_3d
from Models.VGG import vgg16

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

# def extract_vgg_feature (dataset_dir):
#     dataset_name = 'grasp'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     resnet_rgb_extractor = vgg16(pretrained=True).to(device)
#     resnet_rgb_extractor.eval()
#     for param in resnet_rgb_extractor.parameters():
#         param.requires_grad = False
        
#     transform = transforms.Compose([
#                                 transforms.Resize((448,448), interpolation=3),
# #                                 transforms.Resize((224,224), interpolation=3),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                                 ])
        
#     videos_feature_dict = {}
#     video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
#     for video_name in video_name_list:
#         print(video_name+' is being processing.')
        
#         video_dir = join(dataset_dir, video_name)
#         video_frames_dir = join(video_dir, 'frame')
            
#         seq_len = len(os.listdir(video_frames_dir))
#         video_frames_tensor = torch.stack([
#                                 transform(Image.open(join(video_frames_dir, format(frame_idx, '05d')+'.jpg'))) 
#                                 for frame_idx in range(seq_len)], dim=0)  #1 x seq_len x 3 x 448 x 448
#         video_frames_tensor = video_frames_tensor.unsqueeze(0)
        
#         # get video's resnet feature maps
#         video_feature = []
#         for idx in range(seq_len):
#             batched_frames = video_frames_tensor[:,idx,:,:,:].to(device)   # 1x3x448x448
#             with torch.no_grad():
#                 batched_feature = resnet_rgb_extractor(batched_frames) # 1xCxWxH
#             video_feature.append(batched_feature) 
#         video_feature = torch.cat(video_feature, 0).cpu()   # seq_len x C x W x H
# #         torch.save(video_feature, join(video_dir, 'feature', 'resnet101_rgb_pretrain_conv5.pt'))
        
#         videos_feature_dict[video_name] = video_feature
#         print(video_feature.size())
#         print(video_name+' is finished.')
#         utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/'+dataset_name+'_vgg16_rgb_pool5/', 
#                                           (240,360), video_name, dataset_dir)
#     return videos_feature_dict

# def extract_3d_feature (dataset_dir):
#     dataset_name = 'grasp'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     resnet_rgb_extractor = resnet_3d.resnet34(sample_size=112, sample_duration=16, shortcut_type='A', 
#                                               num_classes=400, last_fc=True, output='conv4')
#     model_dict = resnet_rgb_extractor.state_dict()
# #     print(list(model_dict.keys())[:5])
#     checkpoint = torch.load('Models/resnet34_3d_kinetics.pth')
#     pretrain_dict = checkpoint['state_dict']
# #     print(list(pretrain_dict.keys())[:5])
# #     print([k for k in pretrain_dict.keys() if 'module.' not in k])
#     pretrain_dict = {k[7:]: v for k,v in pretrain_dict.items() if k in model_dict}
#     model_dict.update(pretrain_dict)
#     resnet_rgb_extractor.load_state_dict(model_dict)
# #     resnet_rgb_extractor.load_state_dict(torch.load('Models/resnet34_3d_kinetics.pth')['state_dict'])
#     for param in resnet_rgb_extractor.parameters():
#         param.requires_grad = False
#     resnet_rgb_extractor.to(device)
#     resnet_rgb_extractor.eval()
        
#     transform = transforms.Compose([
#                                 transforms.Resize((224,224), interpolation=3),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# #                                 transforms.Normalize(mean=[114.7748, 107.7354, 99.4750], std=[1,1,1])
#                                 ])
        
#     videos_feature_dict = {}
#     video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
#     for video_name in video_name_list[0:5]:
#         print(video_name+' is being processing.')
        
#         video_dir = join(dataset_dir, video_name)
#         video_frames_dir = join(video_dir, 'frame')
            
#         seq_len = len(os.listdir(video_frames_dir))
#         video_frames_tensor = torch.stack([
#                                 transform(Image.open(join(video_frames_dir, format(frame_idx, '05d')+'.jpg'))) 
#                                 for frame_idx in range(seq_len)], dim=0)  #1 x seq_len x 3 x 448 x 448
#         video_frames_tensor = video_frames_tensor.unsqueeze(0)
        
#         # get video's resnet feature maps
#         video_feature = []
#         for idx in range(0, seq_len-15):
#             batched_frames = video_frames_tensor[:,idx:idx+16,:,:,:].transpose(1,2).to(device)   # 1x3x16x224x224
#             with torch.no_grad():
#                 batched_feature = resnet_rgb_extractor(batched_frames) # 1x512x1x7x7
#                 h,w = batched_feature.size()[3:]
#                 batched_feature = batched_feature.view(1,-1,h,w)    # 1xCxWxH
#             video_feature.append(batched_feature) 
#         video_feature = torch.cat(video_feature, 0).cpu()   # seq_len x C x W x H
#         torch.save(video_feature, join(video_dir, 'feature', 'c3d_rgb_pretrain_conv5b.pt'))
        
#         videos_feature_dict[video_name] = video_feature
#         print(video_feature.size())
#         print(video_name+' is finished.')
#         utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/'+dataset_name+'_c3d_rgb_pretrain_conv5b/', 
#                                           (240,360), video_name, dataset_dir)
#     return videos_feature_dict

# def extract_c3d_feature (dataset_dir):
#     dataset_name = 'grasp'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     c3d_extractor = C3D(output='pool5').to(device)
#     c3d_dict = c3d_extractor.state_dict()
#     pretrained_dict = torch.load('Models/c3d_rgb_pretrain.pickle')
#     pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in c3d_dict}
# #     c3d_dict.update(pretrained_dict)
#     c3d_extractor.load_state_dict(pretrained_dict)
#     for param in c3d_extractor.parameters():
#         param.requires_grad = False
#     c3d_extractor.eval()
        
#     transform = transforms.Compose([
#                                 transforms.Resize((224,224), interpolation=3),
#                                 transforms.ToTensor(),
# #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1,1,1])
#                                 ])
        
#     videos_feature_dict = {}
#     video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
#     for video_name in video_name_list[0:5]:
#         print(video_name+' is being processing.')
        
#         video_dir = join(dataset_dir, video_name)
#         video_frames_dir = join(video_dir, 'frame')
            
#         seq_len = len(os.listdir(video_frames_dir))
#         video_frames_tensor = torch.stack([
#                                 transform(Image.open(join(video_frames_dir, format(frame_idx, '05d')+'.jpg'))) 
#                                 for frame_idx in range(seq_len)], dim=0)  #1 x seq_len x 3 x 448 x 448
#         video_frames_tensor = video_frames_tensor.unsqueeze(0)
        
#         # get video's resnet feature maps
#         video_feature = []
#         for idx in range(0, seq_len-15):
#             batched_frames = video_frames_tensor[:,idx:idx+16,:,:,:].transpose(1,2).to(device)   # 1x3x16x224x224
#             with torch.no_grad():
#                 batched_feature = c3d_extractor(batched_frames) # 1x512x2x14x14
#                 h,w = batched_feature.size()[3:]
#                 batched_feature = batched_feature.view(1,-1,h,w)    # 1xCxWxH
#             video_feature.append(batched_feature) 
#         video_feature = torch.cat(video_feature, 0).cpu()   # seq_len x C x W x H
#         torch.save(video_feature, join(video_dir, 'feature', 'c3d_rgb_pretrain_conv5b.pt'))
        
#         videos_)feature_dict[video_name] = video_feature
#         print(video_feature.size())
#         print(video_name+' is finished.')
#         utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/'+dataset_name+'_c3d_rgb_pretrain_conv5b/', 
#                                           (240,360), video_name, dataset_dir)
#     return videos_feature_dict

def extract_save_rgb_feature (dataset_dir):
    dataset_name = 'grasp'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet_rgb_extractor = resnet101(pretrained=True, channel=3, output='conv5').to(device)
    pretrained_model = torch.load('Models/ResNet101_rgb_pretrain.pth.tar')
    resnet_rgb_extractor.load_state_dict(pretrained_model['state_dict'])
    for param in resnet_rgb_extractor.parameters():
        param.requires_grad = False
        
    transform = transforms.Compose([
#                                 transforms.Resize((448,448), interpolation=3),
                                transforms.Resize((224,224), interpolation=3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        
    videos_feature_dict = {}
    video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
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
        for idx in range(seq_len):
            batched_frames = video_frames_tensor[:,idx,:,:,:].to(device)   # 1x3x448x448
            with torch.no_grad():
                batched_feature = resnet_rgb_extractor(batched_frames) # 1xCxWxH
            video_feature.append(batched_feature) 
        video_feature = torch.cat(video_feature, 0).cpu()   # seq_len x C x W x H
        torch.save(video_feature, join(video_dir, 'feature', 'resnet101_rgb_pretrain_conv5.pt'))
        
        videos_feature_dict[video_name] = video_feature
        print(video_feature.size())
        print(video_name+' is finished.')
        utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/'+dataset_name+'_resnet101_rgb_pretrain_conv5/', 
                                          (240,360), video_name, dataset_dir)
    return videos_feature_dict

def extract_save_flow_feature (dataset_dir):
    dataset_name = 'grasp'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet_flow_extractor = resnet101(pretrained=True, channel=20, output='conv4').to(device)
    pretrained_model = torch.load('Models/ResNet101_flow_pretrain.pth.tar')
    resnet_flow_extractor.load_state_dict(pretrained_model['state_dict'])
    for param in resnet_flow_extractor.parameters():
        param.requires_grad = False
        
    transform = transforms.Compose([
                                transforms.Resize((224,224), interpolation=3),
                                transforms.ToTensor(),
                                ])
        
    videos_feature_dict = {}
    video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
    for video_name in video_name_list:
        print(video_name+' is being processing.')
        
        video_dir = join(dataset_dir, video_name)
        video_frames_tensor = []
        
        seq_len = int( len(os.listdir(join(video_dir, 'flow'))) / 3 )
        for frame_idx in range(1, seq_len+1):
            video_frames_tensor.append(
                transform(Image.open(join(video_dir, 'flow', 'flow_x_'+format(frame_idx, '05d')+'.jpg'))))
            video_frames_tensor.append(
                transform(Image.open(join(video_dir, 'flow', 'flow_y_'+format(frame_idx, '05d')+'.jpg'))))
        video_frames_tensor = torch.cat(video_frames_tensor, 0).unsqueeze(0)    #1 x seq_lenx2 x 448 x 448
        
        # get video's resnet feature maps
        video_feature = []
        for idx in range(seq_len-9):
            batched_frames = video_frames_tensor[:,2*idx:2*(idx+10),:,:].to(device)   # 1x20x448x448
            with torch.no_grad():
                batched_feature = resnet_flow_extractor(batched_frames) # 1xCxWxH
            video_feature.append(batched_feature) 
        video_feature = torch.cat(video_feature, 0).cpu()   # seq_len-9 x 256 x 28 x 28
        torch.save(video_feature, join(video_dir, 'feature', 'resnet101_flow_10_pretrain_conv4.pt'))
#         if isfile(join(video_dir, 'feature', 'resnet_flow_10_pretrain_conv5.pt')):
#             os.system('mv '+join(video_dir, 'feature', 'resnet_flow_10_pretrain_conv5.pt'))
        
        videos_feature_dict[video_name] = video_feature
        print(video_feature.size())
        print(video_name+' is finished.')
        utility.save_featmap_heatmaps(video_feature.cpu().data, 'results/'+dataset_name+'_resnet101_flow_pretrain_conv4/', 
                                          (240,360), video_name, dataset_dir)
    return videos_feature_dict

def get_rgb_feature_dict (dataset_dir, feature_type='resnet101_conv5'):
    videos_feature_dict = {}
    video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
    for video_name in video_name_list:
        video_dir = join(dataset_dir, video_name)
        
        if feature_type == 'resnet101_conv5':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet101_rgb_pretrain_conv5.pt'))
        elif feature_type == 'resnet101_conv4':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet101_rgb_pretrain_conv4.pt'))
        
        videos_feature_dict[video_name] = video_feature
    return videos_feature_dict

def get_flow_feature_dict (dataset_dir, feature_type='resnet101_conv5'):
    videos_feature_dict = {}
    video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
    for video_name in video_name_list:
        video_dir = join(dataset_dir, video_name)
        
        if feature_type == 'resnet101_conv5':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet101_flow_10_pretrain_conv5.pt'))
        elif feature_type == 'resnet101_conv4':
            video_feature = torch.load(join(video_dir, 'feature', 'resnet101_flow_10_pretrain_conv4.pt'))
        
        videos_feature_dict[video_name] = video_feature
#         print(video_name, video_feature.size(0))
    return videos_feature_dict

def del_featmaps (dataset_dir):
    video_name_list = [video_name for video_name in os.listdir(dataset_dir) if '_L' in video_name]
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
            if 'resnet101_rgb_pretrain_conv5_7x7' in file:
                print("delete: ", file)
                os.system('rm '+join(video_feature_dir, file))

# ============================================================================= #
#                           Dataset for One Video                               #
# ============================================================================= #  
def video_sample (video_tensor, rand_idx_list):
    sampled_video_tensor = torch.stack([video_tensor[idx,:,:,:] for idx in rand_idx_list], dim=0)
    return sampled_video_tensor  

class GraspDataset(Dataset):
    def __init__(self, f_type, video_rgb_feature_dict, video_flow_feature_dict, video_name_list, seg_sample=None):
        self.seg_sample = seg_sample
        self.video_rgb_feature_dict = video_rgb_feature_dict
        self.video_flow_feature_dict = video_flow_feature_dict
        self.video_name_list = video_name_list
        self.f_type = f_type

    def __len__(self):
        return len(self.video_name_list)
        
    def __getitem__(self, index):
        video_name = self.video_name_list[index]
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
            video_tensor = torch.cat([video_rgb_tensor, video_flow_tensor], dim=1) ##num_seg x 4096 x 14 x 14
              
        sample = {'name': video_name, 'video': video_tensor, 'sampled_index': rand_rgb_idx_list}
        return sample
        
# ============================================================================= #
#                            Dataset for One Pair                               #
# ============================================================================= # 
class GraspDataset_Pair(Dataset):
    def __init__(self, f_type, video_rgb_feature_dict, video_flow_feature_dict, pairs_dict, seg_sample=None):
        self.seg_sample = seg_sample
        self.video_rgb_feature_dict = video_rgb_feature_dict
        self.video_flow_feature_dict = video_flow_feature_dict
        self.f_type = f_type
        
        self.pairs_dict = pairs_dict
        self.pairs_list = list(self.pairs_dict.keys())

    def __len__(self):
        return len(self.pairs_dict)
        
    def __getitem__(self, index):
        v1_name, v2_name = self.pairs_list[index]
        v1_sample = self.read_one_video(v1_name)
        v2_sample = self.read_one_video(v2_name)
        label = torch.Tensor([ self.pairs_dict[(v1_name, v2_name)] ])
        
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
#     extract_save_flow_feature('../dataset/InfantsGrasping/InfantsGrasping_480x720')
#     extract_save_rgb_feature('../dataset/InfantsGrasping/InfantsGrasping_480x720')
    del_featmaps('../dataset/InfantsGrasping/InfantsGrasping_480x720')
