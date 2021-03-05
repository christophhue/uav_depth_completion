#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:07:52 2018

@author: norbot
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import data_transform
from PIL import Image, ImageOps
import h5py
import cv2

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
imagenet_eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
imagenet_eigvec = np.array([[-0.5675,  0.7192,  0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948,  0.4203]], dtype=np.float32)

EXTENSIONS = ['.jpg', '.png', '.tif','.JPG']

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

class NyuDepthDataset(Dataset):
    # nyu depth dataset 
    def __init__(self, root_dir, split, n_sample=200, input_format = 'img'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.input_format = input_format
        self.n_sample = n_sample
        self.images_root = r"C:\Users\student\Documents\Drone_Dataset_update\Img"
        self.labels_root = r"C:\Users\student\Documents\Drone_Dataset_update\TrueDepth_GT"

        self.images_root = os.path.join(self.images_root, split)
        self.labels_root = os.path.join(self.labels_root, split)
        #self.images_root = os.path.join("/Volumes/Elements_mac/depth_prediction/Drone_subset/", 'Flug2/')
        #self.labels_root = os.path.join("/Volumes/Elements_mac/depth_prediction/Drone_subset/GT", 'Flug2/')
        # self.images_root += split
        # self.labels_root += split

        print(self.images_root)
        # self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in
                          fn if is_image(f)]
        self.filenames.sort()
        # print()

        # [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        # self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in
                            fn if is_image(f)]
        self.filenamesGt.sort()
    
    def __len__(self):
        return len(self.filenamesGt)

    def __getitem__(self, idx):
        # read input image
        filename = self.filenames[idx]
        filenameGt = self.filenamesGt[idx]
        rgb_image = Image.open(filename).convert('RGB')
        # with open(image_path_city(self.images_root, filename), 'rb') as f:
        #    rgb_image = load_image(f).convert('RGB')
        # with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
        # depth_image = io.imread(f)
        depth_image = cv2.imread(filenameGt, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
        depth_image = Image.fromarray(depth_image.astype('float32'), mode='F')
        
        _s = np.random.uniform(1.0, 1.5)
        s = np.int(240*_s)
        s = (352, 512)
        degree = np.random.uniform(-5.0, 5.0)
        if self.split == 'train':
            tRgb = data_transform.Compose([transforms.Resize(s),
                                           #transforms.CenterCrop((352, 512)),
                                           data_transform.Rotation(degree),
                                           transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4),
#                                           data_transform.Lighting(0.1, imagenet_eigval, imagenet_eigvec)])
                                           transforms.CenterCrop((352, 512)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                           transforms.ToPILImage()])

            tDepth = data_transform.Compose([transforms.Resize(s),
                                             data_transform.Rotation(degree),
                                             transforms.CenterCrop((352, 512))])
            rgb_image = tRgb(rgb_image)
            depth_image = tDepth(depth_image)
            if np.random.uniform()<0.5:
                rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
                depth_image = depth_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            rgb_image = transforms.ToTensor()(rgb_image)
            if self.input_format == 'img':
                depth_image = transforms.ToTensor()(depth_image)
            else:
                depth_image = data_transform.ToTensor()(depth_image)

            ##depth preprocessing
            depth_image = np.asarray(depth_image, dtype=np.float32)
            sparse_depth_s = np.zeros(depth_image.shape)
            #print(sparse_depth_s.shape)
            #print(depth_image.shape)
            mask_l = depth_image > 0
            #t = torch.from_numpy(mask_l.astype(np.bool))
            mask_keep = np.bitwise_and(mask_l, depth_image <= 500)
            sparse_depth_s[mask_keep] = depth_image[mask_keep]
            depth_image = sparse_depth_s
            # print(depth_image.max())

            max_depth = max(depth_image.max(), 1.0)
            depth_image = (10 / max_depth) * depth_image

            depth_image = depth_image/_s
            depth_image = torch.from_numpy(depth_image.astype(np.float32))
            sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)



        elif self.split == 'val':
            tRgb = data_transform.Compose([transforms.Resize((365,547)),
                                           transforms.CenterCrop(( 352,  512)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                           transforms.ToPILImage()])

            tDepth = data_transform.Compose([transforms.Resize((365,547)),
                                             transforms.CenterCrop(( 352,  512))])
            rgb_image = tRgb(rgb_image)
            rgb_raw = tDepth(rgb_image)
            depth_image = tDepth(depth_image)
            rgb_image = transforms.ToTensor()(rgb_image)



            ### exclude points with depth > 500m ####
            depth_image = np.asarray(depth_image, dtype=np.float32)
            sparse_depth = np.zeros(depth_image.shape)
            mask_l = depth_image > 0
            mask_keep = np.bitwise_and(mask_l, depth_image <= 500)
            sparse_depth[mask_keep] = depth_image[mask_keep]
            depth_image = sparse_depth
            max_depth = max(depth_image.max(), 1.0)
            depth_image = (10 / max_depth) * depth_image
            depth_image = torch.from_numpy(depth_image.astype(np.float32)).unsqueeze(0)
            #print("Val Depth Tensor", depth_image.shape)

            # if self.input_format == 'img':
            #     depth_image = transforms.ToTensor()(depth_image)
            # else:
            #     depth_image = data_transform.ToTensor()(Image.fromarray(depth_image))

            sparse_image = self.createSparseDepthImage(depth_image, self.n_sample)
            rgbd_image = torch.cat((rgb_image, sparse_image), 0)
            rgb_raw = transforms.ToTensor()(rgb_raw)

        sample = {'rgbd': rgbd_image, 'depth': depth_image, 'raw_rgb': rgb_raw }
        return sample
    
    def createSparseDepthImage(self, depth_image, n_sample):
        random_mask = torch.zeros(1, depth_image.shape[1], depth_image.shape[2])
        n_pixels = depth_image.shape[1] * depth_image.shape[2]
        n_valid_pixels = torch.sum(depth_image>0.0001)
#        print('===> number of total pixels is: %d\n' % n_pixels)
#        print('===> number of total valid pixels is: %d\n' % n_valid_pixels)
        perc_sample = n_sample/n_pixels
        random_mask = torch.bernoulli(torch.ones_like(random_mask)*perc_sample)
        sparse_depth = torch.mul(depth_image, random_mask)
        return sparse_depth

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
    #    print (f.keys())
        rgb = f['rgb'][:].transpose(1,2,0)
        depth = f['depth'][:]
        return (rgb, depth)


def show_img(image):
    """Show image"""
    plt.imshow(image)


if __name__ == '__main__':
    nyudepth_dataset = NyuDepthDataset(
        root_dir='.',
        split='train',
        n_sample=500,
        input_format='_img')
    print(len(nyudepth_dataset))
    dataloader = torch.utils.data.DataLoader(
        nyudepth_dataset, batch_size=3, shuffle=True,
        num_workers=1, pin_memory=False)

    for batch_idx, (sample) in enumerate(dataloader):
        print("data", sample['depth'].shape)
        # print("depth",target.shape)
        rgb = transforms.ToPILImage()(sample['rgbd'][0:3, :, :].squeeze(0))
        depth = sample['depth'].squeeze(0).squeeze(0).numpy()
        #sparse_depth = transforms.ToPILImage()(sample['rgbd'][3, :, :].squeeze(0))
        #depth_mask = transforms.ToPILImage()(torch.sign(sample['depth']))
        #sparse_depth_mask = transforms.ToPILImage()(sample['rgbd'][3, :, :].unsqueeze(0).sign())
        print(sample['rgbd'][0:3, :, :])
        #invalid_depth = torch.sum(sample['rgbd'][3, :, :].unsqueeze(0).sign() < 0)
        #print(invalid_depth)
        #im = tensor2im(sparse_rgb)
        #im2 = tensor2im(sparse_depth)
        plt.imshow(depth,cmap='jet')
        #plt.clf()
        #plt.imshow(im2)
        plt.show()
        if batch_idx == 3:
            break
