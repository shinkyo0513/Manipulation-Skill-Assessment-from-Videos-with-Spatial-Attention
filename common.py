import torch

import os
from os.path import join

from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.ImageShow import voxel_tensor_to_np, overlap_maps_on_voxel_np, img_np_show

##################################################
#                  Train & Test                  #
##################################################
def train(dataloader, model, criterion, optimizer, epoch, device):
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
        running_acc += torch.nonzero((label*(v1_score-v2_score))
                                     > 0).size(0) / v1_tensor.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    print(f'Train: Epoch {epoch}, Loss:{epoch_loss:.4f}, Acc:{epoch_acc:.4f}')


def test(dataloader, pairs_dict, model, criterion, epoch, device):
    model.eval()

    videos_score = {}
    videos_satt = {}
    for batch_idx, sample in enumerate(tqdm(dataloader)):
        video_name = sample['name']
        v_tensor = sample['video'].to(device)
        sampled_idx_list = sample['sampled_index']

        with torch.no_grad():
            v_score, v_satt, v_tatt = model(v_tensor)
        for i in range(v_tensor.size(0)):
            videos_score[video_name[i]] = v_score[i].unsqueeze(0)
            videos_satt[video_name[i]] = v_satt[i].unsqueeze(0)

    running_loss = 0.0
    running_acc = 0.0

    pairs_list = list(pairs_dict.keys())
    for v1_name, v2_name in pairs_list:
        v1_score = videos_score[v1_name]
        v2_score = videos_score[v2_name]
        v1_satt = videos_satt[v1_name]
        v2_satt = videos_satt[v2_name]
        label = torch.Tensor([pairs_dict[(v1_name, v2_name)]]).to(device)

        loss = criterion(v1_score, v2_score, label)

        running_loss += loss.item()
        running_acc += torch.nonzero((label*(v1_score-v2_score)) > 0).size(0)

    epoch_loss = running_loss / len(pairs_list)
    epoch_acc = running_acc / len(pairs_list)

    print(f'Test: Epoch {epoch}, Loss:{epoch_loss:.4f}, Acc:{epoch_acc:.4f}')

    return epoch_loss, epoch_acc


def save_best_result(dataloader, test_videos, model, device, best_acc, save_label):
    model.eval()

    best_checkpoint = {'state_dict': model.state_dict(), 'best_acc': best_acc}
    ckpt_dir = join('checkpoints', save_label)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(best_checkpoint, join(ckpt_dir, 'best_checkpoint.pth.tar'))

    videos_score = {}
    htmp_dir = join('heatmaps', save_label)
    os.makedirs(htmp_dir, exist_ok=True)
    for batch_idx, sample in enumerate(dataloader):
        video_name = sample['name']
        v_tensor = sample['video'].to(device)
        sampled_idx_list = sample['sampled_index']

        with torch.no_grad():
            v_score, v_satt, v_tatt = model(v_tensor)

        for i in range(v_tensor.size(0)):
            pred_score = v_score[i].item()
            videos_score[video_name[i]] = pred_score

            if video_name[i] in test_videos:
                plot_video_heatmaps(v_tensor[i], v_satt[i], title=f'{video_name[i]} Pred:{pred_score:.4f}', 
                                    save_path=join(htmp_dir, f'{video_name[i]}.jpg'))

    score_dir = join('pred_scores', save_label)
    os.makedirs(score_dir, exist_ok=True)
    with open(join(score_dir, 'scores.txt'), 'w') as f:
        score = v_score.detach().cpu().item()
        f.writeline(f'{score:.4f}\n')
    f.close()

def plot_video_heatmaps (video_tensor, heatmap, title=None, save_path=None, save_separately=False):
    # video_tensor: 3xLx112x112
    # heatmap: 1xLx112x112
    num_timesteps = video_tensor.shape[1]
    assert num_timesteps == heatmap.shape[1]

    video_imgs = voxel_tensor_to_np(video_tensor)  # np, 0~1, 3xLx112x112
    video_imgs_uint = np.uint8(video_imgs * 255)

    if torch.is_tensor(heatmap):
        heatmap = heatmap.squeeze(0).numpy()   # np, 0~1, Lx112x112
    else:
        heatmap = heatmap[0]
        
    overlaps = overlap_maps_on_voxel_np(video_imgs, heatmap)   # np, 0~1, 3xLx112x112
    overlaps_uint = np.uint8(overlaps * 255)

    if save_separately and save_path != None:
        separate_save_dir = os.path.splitext(save_path)[0]
        os.makedirs(separate_save_dir, exist_ok=True)

    # save plot imgs, explanation heatmap
    num_subline = 2
    num_row = num_subline * ( (num_timesteps-1) // 8 + 1 )
    plt.clf()
    fig = plt.figure(figsize=(16,num_row*2))
    for i in range(num_timesteps):
        plt.subplot(num_row, 8, (i//8)*8*num_subline+i%8+1)
        img_np_show(video_imgs_uint[:,i])
        plt.title(i, fontsize=8)

        plt.subplot(num_row, 8, (i//8)*8*num_subline+i%8+8+1)
        img_np_show(overlaps_uint[:,i])

        if save_separately:
            video_img = Image.fromarray(video_imgs_uint[:,i].transpose(1,2,0))
            video_img.save(os.path.join(separate_save_dir, f'img_{i}.jpg'))
            exp_img = Image.fromarray(overlaps_uint[:,i].transpose(1,2,0))
            exp_img.save(os.path.join(separate_save_dir, f'exp_{i}.jpg'))

    if title != None:
        fig.suptitle(title, fontsize=14)

    if save_path != None:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)

        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    plt.close(fig)

# # batched_heatmaps: batch_size x seq_len x 1 x 7 x 7
# def save_heatmaps(batched_inputs, batched_heatmaps, save_dir, size, video_name, rand_idx_list, t_att):
#     batch_size = batched_heatmaps.size(0)
#     seq_len = batched_heatmaps.size(1)

#     for batch_offset in range(batch_size):
#         att_save_dir = join(save_dir, video_name[batch_offset])
#         os.makedirs(att_save_dir, exist_ok=True)

#         dataset_name, video_name = video_name[batch_offset].split('_')
#         dataset_dir = dataset_dir = join(
#             '../dataset', dataset_name, dataset_name+'_640x480')
#         ori_frames_dir = join(dataset_dir, video_name, 'frame')

#         for seq_idx in range(seq_len):
#             frame_idx = int(rand_idx_list[seq_idx][batch_offset].item())

#             heatmap = batched_heatmaps[batch_offset, seq_idx, 0, :, :]
#             heatmap = (heatmap-heatmap.min()) / (heatmap.max()-heatmap.min())
#             heatmap = np.array(heatmap*255.0).astype(np.uint8)
#             heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#             heatmap = cv2.resize(heatmap, size)

#             ori_frame = cv2.imread(
#                 join(ori_frames_dir, format(frame_idx, '05d')+'.jpg'))
#             ori_frame = cv2.resize(ori_frame, size)

#             comb = cv2.addWeighted(ori_frame, 0.6, heatmap, 0.4, 0)
#             t_att_value = t_att[batch_offset, seq_idx].item()
#             pic_save_dir = join(att_save_dir, format(
#                 frame_idx, '05d')+'_'+format(t_att_value, '.2f')+'.jpg')
#             cv2.imwrite(pic_save_dir, comb)

