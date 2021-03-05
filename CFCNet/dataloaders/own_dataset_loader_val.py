#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:07:52 2018

@author: norbot
"""

from __future__ import print_function, division
import os
import torch
import cv2
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import img_as_float
from torchvision import transforms, utils
import dataloaders.data_transform as data_transform
import dataloaders.transforms as cfctransforms
from skimage import io

from PIL import Image, ImageOps
import h5py
from numba import jit, njit

EXTENSIONS = ['.jpg', '.png', '.tif', '.JPG']


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')


def image_path_city(root, name):
    return os.path.join(root, f'{name}')


@njit
def fast_ops(arr, arr1, arr2, arr3):
    arr = np.concatenate((arr, np.expand_dims(arr1, axis=2)), axis=2)
    arr = np.concatenate((arr, arr2), axis=2)
    arr = np.concatenate((arr, np.expand_dims(arr3, axis=2)), axis=2)
    return arr


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
imagenet_eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
imagenet_eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203]], dtype=np.float32)


class OwnDataset(Dataset):
    # nyu depth dataset 
    def __init__(self, images_root="C:\Users\student\Documents\Drone_Dataset_update\Img",
                 depth_root="C:\Users\student\Documents\Drone_Dataset_update\RefinedDepth_GT", split='train',
                 sparsifier=None, modality='rgb', input_format='img'):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): train/val/test split
            sparsifier: sparsifier method (Stereo, ORB, etc.)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.rgbd_frame = pd.read_csv(csv_file)
        # self.root_dir = root_dir
        self.split = split
        self.input_format = input_format
        self.sparsifier = sparsifier
        self.modality = modality

        # to debug set paths yourself
        # self.images_root = os.path.join("/Volumes/Elements_mac/depth_prediction/depth_data/Forumsplatz_converted/", 'Flug2/')
        # self.labels_root = os.path.join("/Volumes/Elements_mac/depth_prediction/depth_data/Forumsplatz_Depth", 'Flug2/')
        # self.images_root = images_root
        # self.labels_root = depth_root
        #self.images_root = r"C:\Users\student\Documents\Drone_Dataset_update\Img"
        #self.labels_root = r"C:\Users\student\Documents\Drone_Dataset_update\RefinedDepth_GT"
        self.images_root = images_root
        self.labels_root = depth_root

        self.images_root = os.path.join(self.images_root, split)
        self.labels_root = os.path.join(self.labels_root, split)
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
        self.listoffiles = []
        self.listoffilesGt = []
    
        print("Number of images", len(self.filenames))
        print("Number of Depthmaps", len(self.filenamesGt))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # read input image
        filename = self.filenames[idx]
        filenameGt = self.filenamesGt[idx]
        rgb_image = Image.open(filename).convert('RGB')
        depth_image = cv2.imread(filenameGt, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))

        
        if depth_image.ndim < 2:
            print(depth_image.shape)
            print(filenameGt)
        _s = np.random.uniform(1.0, 1.5)
        depth_image = depth_image / _s
        s = (np.int(365 * _s), np.int(547 * _s))
        depth_image = np.asarray(cv2.resize(depth_image, dsize=(s[1], s[0]), interpolation=cv2.INTER_NEAREST),
                                 dtype=np.float32)
        # s = (912,608)
        degree = np.random.uniform(-5.0, 5.0)
        do_flip = np.random.uniform(0.0, 1.0)
        if self.split == 'train':
            tRgb = data_transform.Compose([  # transforms.functional.crop(130,10,1368,912),

                transforms.Resize(s),
                data_transform.Rotation(degree),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

                #                                           data_transform.Lighting(0.1, imagenet_eigval, imagenet_eigvec)])
                # transforms.CenterCrop((228*1, 304*1)),
                # transforms.CenterCrop((365,547)),
                transforms.CenterCrop((352, 512)),
                transforms.ToTensor(),
            ])

            tDepth = data_transform.Compose([  # transforms.functional.crop(130,10,1368,912),
                cfctransforms.Rotate(degree),
                cfctransforms.CenterCrop((352, 512)),
            ])

            rgb_image = tRgb(rgb_image)

            depth_image = tDepth(depth_image)
            depth_image = np.asarray(depth_image, dtype=np.float32)

            ### exclude points with depth > 500m ####
            sparse_depth = np.zeros(depth_image.shape)
            mask_l = depth_image > 0
            mask_keep = np.bitwise_and(mask_l, depth_image <= 500)
            sparse_depth[mask_keep] = depth_image[mask_keep]
            depth_image = sparse_depth
            # print(depth_image.max())

            # depth_image = scale(depth_image, out_range=(0.01, 1))
            max_depth = max(depth_image.max(), 1.0)
            depth_image = (10 / max_depth) * depth_image
        if self.split == 'val' or self.split == 'test':
            s = (365, 547)
            depth_image = np.asarray(cv2.resize(depth_image, dsize=(s[1], s[0]), interpolation=cv2.INTER_NEAREST),
                                     dtype=np.float32)

            tRgb = data_transform.Compose([ 
                transforms.Resize(s),
                transforms.CenterCrop((352, 512)),
                transforms.ToTensor(),
            ])

            tDepth = data_transform.Compose([  
                cfctransforms.Resize(1.0),
                cfctransforms.CenterCrop((352, 512)),
            ])

            rgb_image = tRgb(rgb_image)
            # depth_image = transforms.functional.crop(depth_image, 130, 10, 548, 821)
            depth_image = tDepth(depth_image)
            # print(depth_image.shape)
            depth_image = np.asarray(depth_image, dtype=np.float32)

            ### exclude points with depth > 500m ####
            sparse_depth = np.zeros(depth_image.shape)
            mask_l = depth_image > 0
            mask_keep = np.bitwise_and(mask_l, depth_image <= 500)
            sparse_depth[mask_keep] = depth_image[mask_keep]
            depth_image = sparse_depth


            #print("max", depth_image.max())
            # depth_image = scale(depth_image, out_range=(0.01, 1))
            max_depth = max(depth_image.max(), 1.0)
            depth_image = (10 / max_depth) * depth_image
            scale = max_depth / 10

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)

        depth_image = transforms.ToTensor()(depth_image)

        return transforms.ToTensor()(self.create_rgbdm(rgb_image.squeeze(0).numpy().transpose(1, 2, 0),
                                                       depth_image.squeeze(0).numpy())), depth_image, scale

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_sparse_depth_rgb(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            sparse_rgb = np.zeros(rgb.shape)
            sparse_rgb[mask_keep, :] = rgb[mask_keep, :]
            sparse_mask = np.zeros(depth.shape)
            sparse_mask[mask_keep] = 1
            mask_keep = mask_keep.astype(np.uint8)
            return sparse_depth, sparse_rgb, mask_keep

    def create_rgbdm(self, rgb, depth):
        sparse_depth, sparse_rgb, mask = self.create_sparse_depth_rgb(rgb, depth)
        # rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2),axis=2)
        # rgbdm = np.append(rgbd, sparse_rgb, axis=2)
        # rgbdm = np.append(rgbdm, np.expand_dims(mask, axis=2),axis=2)

        return fast_ops(rgb, sparse_depth, sparse_rgb, mask)

    def createSparseDepthImage(self, depth_image, n_sample):
        random_mask = torch.zeros(1, depth_image.size(1), depth_image.size(2))
        n_pixels = depth_image.size(1) * depth_image.size(2)
        n_valid_pixels = torch.sum(depth_image > 0.0001)
        #        print('===> number of total pixels is: %d\n' % n_pixels)
        #        print('===> number of total valid pixels is: %d\n' % n_valid_pixels)
        perc_sample = float(n_sample) / n_valid_pixels.float()
        #        print(random_mask.type())
        #        print(torch.ones_like(random_mask).type())
        #        print(perc_sample)
        random_mask = torch.bernoulli((torch.ones_like(random_mask) * perc_sample))
        sparse_depth = torch.mul(depth_image, random_mask)
        return sparse_depth

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
        #    print (f.keys())
        rgb = f['rgb'][:].transpose(1, 2, 0)
        depth = f['depth'][:]
        return (rgb, depth)


def show_img(image):
    """Show image"""
    plt.imshow(image)


def test_load_h5():
    def load_h5(h5_filename):
        f = h5py.File(h5_filename, 'r')
        #    print (f.keys())
        rgb = f['rgb'][:].transpose(1, 2, 0)
        depth = f['depth'][:]
        return (rgb, depth)

    file_name = './data/kitti_hdf5/val/11/00466-R.h5'
    rgb_h5, depth_h5 = load_h5(file_name)
    depth_h5 = depth_h5.astype('uint16')
    rgb_image = Image.fromarray(rgb_h5, mode='RGB')
    depth_image = Image.fromarray(depth_h5.astype('uint16'), mode='L')


#    cv2.imwrite('tmp/cv_save_kitti_depth.png', depth_h5)

# test_load_h5()


def test_imgread():
    # train preprocessing   
    kitti_dataset = OwnDataset(csv_file='data/kitti_hdf5/kitti_hdf5_train.csv',
                               root_dir='.',
                               split='train',
                               n_sample=500,
                               input_format='hdf5')
    #    kitti_dataset = KittiDataset(csv_file='data/nyudepth_v2/nyudepthv2_val.csv',
    #                                       root_dir='.',
    #                                       split = 'val',
    #                                       n_sample = 500,
    #                                       input_format='hdf5')
    fig = plt.figure()
    for i in range(len(kitti_dataset)):
        sample = kitti_dataset[i]
        rgb = data_transform.ToPILImage()(sample['rgbd'][0:3, :, :])
        depth = data_transform.ToPILImage()(sample['depth'])
        sparse_depth = data_transform.ToPILImage()(sample['rgbd'][3, :, :].unsqueeze(0))
        depth_mask = data_transform.ToPILImage()(torch.sign(sample['depth']))
        sparse_depth_mask = data_transform.ToPILImage()(sample['rgbd'][3, :, :].unsqueeze(0).sign())
        print(sample['depth'])
        invalid_depth = torch.sum(sample['rgbd'][3, :, :].unsqueeze(0).sign() < 0)
        print(invalid_depth)
        plt.imsave("tmp/plt_save_kitit_depth.png", depth)
        depth.save(("tmp/pil_save_kitti_depth.png"))
        rgb.save("tmp/pil_save_kitti_rgb.png")
        #        print(sample['depth'].size())
        #        print(torch.sign(sample['sparse_depth']))
        ax = plt.subplot(5, 4, i + 1)
        ax.axis('off')
        show_img(rgb)
        ax = plt.subplot(5, 4, i + 5)
        ax.axis('off')
        show_img(depth)
        ax = plt.subplot(5, 4, i + 9)
        ax.axis('off')
        show_img(depth_mask)
        ax = plt.subplot(5, 4, i + 13)
        ax.axis('off')
        show_img(sparse_depth)
        ax = plt.subplot(5, 4, i + 17)
        ax.axis('off')
        show_img(sparse_depth_mask)
        if i == 3:
            plt.show()
            break

# test_imgread()
