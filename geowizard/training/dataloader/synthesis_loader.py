# A reimplemented version in public environments by Xiao Fu and Mu Hu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import cv2

from dataloader.utils import read_text_lines
from dataloader.file_io import *

import pickle
import json
from skimage import io, transform
import numpy as np
import glob
import tqdm
from PIL import Image
import torch
from imgaug import augmenters as iaa
import pandas as pd


class SynthesisDataset(Dataset):
    def __init__(self, data_dir, csv_path, dataset_name = "synthesis",
                 transform=None):
        super(SynthesisDataset, self).__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.img_size = (512, 512)
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.samples = []

        # read data path from csv file with 3 headers (dataset, rgb, depth, normal)
        self.data_info = pd.read_csv(csv_path, header=None)
        self.num_img = len(self.data_info)
        print("Number of train images: ", self.num_img)

        self.samples = []
        for image_idx in range(self.num_img):
            sample = dict()
            sample['rgb'] = os.path.join(self.data_dir, self.data_info.iloc[image_idx, 0])
            sample['depth'] = os.path.join(self.data_dir, self.data_info.iloc[image_idx, 1])
            sample['normal'] = os.path.join(self.data_dir, self.data_info.iloc[image_idx, 2])
            if self.dataset_name == "photoface":
                sample['mask'] = os.path.join(self.data_dir, self.data_info.iloc[image_idx, 3])

            # data augmentation args
            sample['RandomHorizontalFlip'] = 0.4
            sample['distortion_prob'] = 0.05
            sample['to_gray_prob'] = 0.1

            self.samples.append(sample)


    def __getitem__(self, index):
        sample = {}
        sample['domain'] = torch.Tensor([1., 0., 0.]) # indoor
        H, W = self.img_size
        try:
            sample_path = self.samples[index]
            sample['rgb'] = read_img(sample_path['rgb'])  # [H, W, 3]
        except Exception as e:
            print("Error at index: ", sample_path['rgb'])
            sample_path = self.samples[index + 1]
            sample['rgb'] = read_img(sample_path['rgb'])
            
        if self.dataset_name == "photoface":
            sample['depth'], sample['normal'], sample["mask"] = read_photoface_dataset(sample_path['depth'], sample_path['normal'], sample_path['mask'])
        else:
            sample['depth'], sample['normal'], sample["mask"] = read_depth_normal_synthesis(sample_path['depth'], sample_path['normal'])

        H_ori, W_ori = sample['rgb'].shape[:2]

        # 1. Random Crop
        if H_ori >= H and W_ori >= W:
            H_start, W_start = np.random.randint(0, H_ori-H+1), np.random.randint(0, W_ori-W+1)
            sample['rgb'] = sample['rgb'][H_start:H_start + H, W_start:W_start + W]
            sample['depth'] = sample['depth'][H_start:H_start + H, W_start:W_start + W]
            sample['normal'] = sample['normal'][H_start:H_start + H, W_start:W_start + W]

        # 2. Random Horizontal Flip
        if np.random.random() < sample_path['RandomHorizontalFlip']:
            sample['rgb'] = np.copy(np.fliplr(sample['rgb']))
            sample['depth'] = np.copy(np.fliplr(sample['depth']))
            sample['normal'] = np.copy(np.fliplr(sample['normal']))
            sample['normal'][:,:,0] *= -1.

        # 3. Photometric Distortion
        to_gray_prob = sample_path['to_gray_prob']
        distortion_prob = sample_path['distortion_prob']
        brightness_beta = np.random.uniform(-32, 32)
        contrast_alpha = np.random.uniform(0.5, 1.5)
        saturate_alpha = np.random.uniform(0.5, 1.5)
        rand_hue = np.random.randint(-18, 18)

        brightness_do = np.random.random() < distortion_prob
        contrast_do = np.random.random() < distortion_prob
        saturate_do = np.random.random() < distortion_prob
        rand_hue_do = np.random.random() < distortion_prob

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = 0 if np.random.random() > 0.5 else 1
        if np.random.random() < to_gray_prob:
            sample['rgb'] = iaa.Grayscale(alpha=(0.8, 1.0))(image=sample['rgb'])
        else:
            # random brightness
            if brightness_do:
                alpha, beta = 1.0, brightness_beta
                sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

            if mode == 0:
                if contrast_do:
                    alpha, beta = contrast_alpha, 0.0
                    sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

            # random saturation
            if saturate_do:
                img = cv2.cvtColor(sample['rgb'][:,:,::-1], cv2.COLOR_BGR2HSV)
                alpha, beta = saturate_alpha, 0.0
                img[:,:,1] = np.clip((img[:,:,1].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)
                sample['rgb'] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:,:,::-1]

            # random hue
            if rand_hue_do:
                img = cv2.cvtColor(sample['rgb'][:,:,::-1], cv2.COLOR_BGR2HSV)
                img[:, :, 0] = (img[:, :, 0].astype(int) + rand_hue) % 180
                sample['rgb'] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:,:,::-1]

            # random contrast
            if mode == 1:
                if contrast_do:
                    alpha, beta = contrast_alpha, 0.0
                    sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

        # 4. To Tensor
        sample['rgb'] = (torch.from_numpy(np.transpose(sample['rgb'].copy(), (2, 0, 1))) / 255.) * 2.0 - 1.0  # [3, H, W]
        sample['depth'] = torch.from_numpy(sample['depth'][None].copy())  # [1, H, W]
        sample['normal'] = torch.from_numpy(np.transpose(sample['normal'].copy(), (2, 0, 1)))  # [3, H, W]

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size
