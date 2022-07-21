import cv2
import random
import numpy as np
import torch
import os
from torch.utils import data
import rawpy
import argparse
import albumentations as A
from torch.utils.data import DataLoader
from utils.utils import inv_normalization, normalization, read_image


class TrainDataset(data.Dataset):

    def __init__(self, args, transform):
        super().__init__()
        self.args = args
        self.transform = transform
        self.gt_data = np.load(args.datasets_gt_dir, allow_pickle=True)[()]
        self.noise_data = np.load(args.datasets_noisy_dir, allow_pickle=True)[()]
        # train_list_file = os.path.join(self.args.datasets_dir, self.args.train_list)
        # 如果数据集尚未分割，则进行训练集和测试集的分割
        # if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
        #     files = os.listdir(os.path.join(self.args.datasets_dir, self.args.input_path))
        #     # random.shuffle(files)
        #     # n_train = int(self.args.train_size * len(files))
        #     # train_list = files[:n_train]
        #     # test_list = files[n_train:]
        #     np.savetxt(os.path.join(self.args.datasets_dir, self.args.train_list), np.array(files), fmt='%s')
        #     # np.savetxt(os.path.join(self.args.datasets_dir, self.args.test_list), np.array(test_list), fmt='%s')
        # self.imlist = np.loadtxt(train_list_file, str)
        # self.patch_size = 256

    def __getitem__(self, index):
        # inpath = os.path.join(self.args.datasets_dir, self.args.input_path, str(self.imlist[index]))
        # gt_path = os.path.join(self.args.datasets_dir, self.args.ground_path, str(self.imlist[index])[:-9] + 'gt.dng')
        # grad_path = os.path.join(self.args.datasets_dir, self.args.grad_path,
        #                          str(self.imlist[index])[:-9] + 'grad.dng')
        raw_data_expand_c = self.noise_data[index]
        gt = self.gt_data[index]
        # raw_data_expand_c, height, width = read_image(inpath)
        # gt, height, width = read_image(gt_path)
        # grad, height, width = read_image(grad_path)
        temp = np.dstack((gt, raw_data_expand_c)).astype(np.float32)
        augments = self.transform(image=temp)
        noise = normalization(augments['image'][:, :, 4:8],
                              self.args.black_level, self.args.white_level)

        gt = normalization(augments['image'][:, :, 0:4],
                           self.args.black_level, self.args.white_level)
        noise = torch.from_numpy(np.transpose(noise, (2, 0, 1)))
        gt = torch.from_numpy(np.transpose(gt, (2, 0, 1)))
        return gt.float(), noise.float()

    def __len__(self):
        return len(self.gt_data)


class ValDataset(data.Dataset):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gt_data = np.load(args.valsets_gt_dir, allow_pickle=True)[()]
        self.noise_data = np.load(args.valsets_noisy_dir, allow_pickle=True)[()]
        # self.transform = A.Compose([A.RandomCrop(256, 256, always_apply=True, p=1)])
        # val_list_file = os.path.join(self.args.valsets_dir, self.args.val_list)
        #
        # if not os.path.exists(val_list_file) or os.path.getsize(val_list_file) == 0:
        #     files = os.listdir(os.path.join(self.args.valsets_dir, self.args.input_path))
        #     np.savetxt(os.path.join(self.args.valsets_dir, self.args.val_list), np.array(files), fmt='%s')
        # self.imlist = np.loadtxt(val_list_file, str)
        # self.patch_size = 256

    def __getitem__(self, index):
        # inpath = os.path.join(self.args.valsets_dir, self.args.input_path, str(self.imlist[index]))
        # gt_path = os.path.join(self.args.valsets_dir, self.args.ground_path, str(self.imlist[index])[:-9] + 'gt.dng')
        # grad_path = os.path.join(self.args.valsets_dir, self.args.grad_path,
        #                          str(self.imlist[index])[:-9] + 'grad.dng')
        #
        # raw_data_expand_c, height, width = read_image(inpath)
        # gt, height, width = read_image(gt_path)
        # grad, height, width = read_image(grad_path)
        raw_data_expand_c = self.noise_data[index].astype(np.float32)
        gt = self.gt_data[index].astype(np.float32)

        # temp = np.dstack((gt, raw_data_expand_c)).astype(np.float32)

        # augments = self.transform(image=temp)
        noise = normalization(raw_data_expand_c,
                              self.args.black_level, self.args.white_level)
        gt = normalization(gt,
                           self.args.black_level, self.args.white_level)

        noise = torch.from_numpy(np.transpose(noise, (2, 0, 1)))
        gt = torch.from_numpy(np.transpose(gt, (2, 0, 1)))
        return gt.float(), noise.float()

    def __len__(self):
        return len(self.gt_data)



