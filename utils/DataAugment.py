import cv2
import random
import numpy as np
import torch
import os
from torch.utils import data
import rawpy
import imageio
from utils import read_image, addNoise, imgaug




if __name__ == '__main__':
    npath = r'../dataset/train/noisy'
    gtpath = r'../dataset/train/gt'
    item_height = 256
    item_width = 256
    noiselist = os.listdir(npath)
    noiselist.sort(key=lambda x: int(x[:-10]))
    gtlist = os.listdir(gtpath)
    gtlist.sort(key=lambda x: int(x[:-7]))
    # height, width = rawpy.imread(gtpath + '//' + gtlist[0]).raw_image_visible.shape
    num = 0
    gt_dic = {}
    noise_dic = {}
    # 'gaussian', 'speckle', 's&p'
    for m in range(len(noiselist)):
        noise, height, width = read_image(npath + '//' + noiselist[m])
        gt, h, w = read_image(gtpath + '//' + gtlist[m])
        # gt_dic, noise_dic, num1 = imgaug(gt, noise, height, width, item_height, item_width, num, gt_dic, noise_dic)
        #
        # noisegs = addNoise(gt, mode='gaussian')
        # gt_dicgs, noise_dicgs, num2 = imgaug(gt, noisegs, height, width, item_height, item_width, num1, gt_dic,
        #                                      noise_dic)
        #
        # noisespk = addNoise(gt, mode='speckle')
        # gt_dicspk, noise_dicspk, num3 = imgaug(gt, noisespk, height, width, item_height, item_width, num2, gt_dicgs,
        #                                  noise_dicgs)
        # noisesp = addNoise(gt, mode='s&p')
        # gt_dicsp, noise_dicsp, num4 = imgaug(gt, noisesp, height, width, item_height, item_width, num3, gt_dicspk,
        #                                 noise_dicspk)
        #
        # # noisedsg = addNoise(gt, mode='double_salt_gauss')
        # # gt_dicdsg, noise_dicdsg, num5 = imgaug(gt, noisedsg, height, width, item_height, item_width, num4, gt_dicsp,
        # #                                      noise_dicsp)
        #
        # noisedpg = addNoise(gt, mode='double_poisson_gauss')
        # gt_dicdpg, noise_dicdpg, num6 = imgaug(gt, noisedpg, height, width, item_height, item_width, num4, gt_dicsp,
        #                                        noise_dicsp)
        #
        # gt_dic = gt_dicdpg
        # noise_dic = noise_dicdpg
        # num = num6
        gt_dic, noise_dic, num1 = imgaug(gt, noise, height, width, item_height, item_width, num, gt_dic, noise_dic)
        num = num1
        print("正在处理{}......".format(m))

    print("一共生成了{}对数据".format(num))

        # for i in range(0, int((height / 2) / item_height)):
        #     for j in range(0, int((width / 2) / item_width)):
        #         crop_gt = gt[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
        #         crop_noise = noise[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
        #         gt_dic[num] = crop_gt
        #         noise_dic[num] = crop_noise
        #         num += 1
        #         print(num)


    np.save('../datanpy/train/gt/gt_fine.npy', gt_dic)
    np.save('../datanpy/train/noisy/noise_fine.npy', noise_dic)