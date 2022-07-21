import cv2

from utils import rfft, read_image, normalization, inv_normalization, write_image, write_back_dng
import torch
import numpy as np

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# gt_ = r'../dataset/train/gt/39_gt.dng'
#
# noise_ = r'../dataset/train/noisy/39_noise.dng'
# gt_path = r'H:\中兴比赛\dataset\\' + 'gt39_fft.dng'
# n_path = r'H:\中兴比赛\dataset\\' + 'noise39_fft.dng'
# d_path = r'H:\中兴比赛\dataset\\' + 'de39_fft.dng'
# gt, height, width = read_image(gt_)
# gt = normalization(gt, 1024, 16383)
# gt = torch.from_numpy(np.transpose(
#     gt.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
# # print(gt.shape)
# # img = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
#
# label_fft = torch.fft.fft2(gt)
#
# gt_data = label_fft.detach().numpy().transpose(0, 2, 3, 1)
# gt_data = inv_normalization(gt_data, 1024, 16383)
# result_write_data = write_image(gt_data, height, width)
# write_back_dng(gt_, gt_path, result_write_data)
#
# noise, height, width = read_image(noise_)
# noise = normalization(noise, 1024, 16383)
# noise = torch.from_numpy(np.transpose(
#     noise.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
#
# noise_fft = torch.fft.fft2(noise)
# n_data = noise_fft.detach().numpy().transpose(0, 2, 3, 1)
# n_data = inv_normalization(n_data, 1024, 16383)
# n_data = write_image(n_data , height, width)
# write_back_dng(gt_, n_path, n_data)
#
# de = label_fft - noise_fft
# d_data = de.detach().numpy().transpose(0, 2, 3, 1)
# d_data = inv_normalization(d_data, 1024, 16383)
# d_data = write_image(d_data, height, width)
# write_back_dng(gt_, d_path, d_data)


# -*- coding: utf-8 -*-

def fft_image(noise, gt, band):
    img1 = gt[:, :, band]
    f = np.fft.fft2(img1)

    # 默认结果中心点位置是在左上角,
    # 调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)

    # fft结果是复数, 其绝对值结果是振幅
    fimg1 = np.log(np.abs(fshift))

    # 读取图像
    # cimg = cv.imread(r'E:\cloudremove\result\TSRnet3-Res-fft-kpl_false\cloud\cloud_65.tif', 0)
    cimg1 = noise[:, :, band]
    f = np.fft.fft2(cimg1)
    fshift = np.fft.fftshift(f)
    c_img1 = np.log(np.abs(fshift))

    newdel1 = np.abs(c_img1 - fimg1)

    return [img1, cimg1, fimg1, c_img1, newdel1]


from matplotlib import pyplot as plt

gt_ = r'../dataset/val/gt/61_gt.dng'
noise_ = r'../dataset/val/noisy/61_noise.dng'

gt, height, width = read_image(gt_)
gt = normalization(gt, 1024, 16383)
noise, height1, width1 = read_image(noise_)
noise = normalization(noise, 1024, 16383)
imglist = []
for i in range(4):
    imglist.append(fft_image(noise, gt, i))
# img1, cimg1, fimg1, c_img1, newdel1 = fft_image(noise, gt, 0)
# img2, cimg2, fimg2, c_img2, newdel2 = fft_image(noise, gt, 1)
# img3, cimg3, fimg3, c_img3, newdel3 = fft_image(noise, gt, 2)
# img4, cimg4, fimg4, c_img4, newdel4 = fft_image(noise, gt, 3)

# 展示结果
# plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original')
# plt.axis('off')
# plt.figure(dpi=300, figsize=(24, 8))
# fig, ax = plt.subplots(4, 5)
# ax[0, 0].plot(img1, 'gray')
# ax[0, 0].set_title('gt_1')
#
# ax[0, 1].plot(cimg1, 'gray')  #
# ax[0, 1].set_title('noise_1')
#
# ax[1, 0].plot(t, s, 'b-.')  #
# ax[1, 0].set_title('第3个子图 ')
# ax[1, 1].plot(t, s, 'y:')  # #第1行第1列，本例子中右下角
# ax[1, 1].set_title('第4个子图 by 桔子code')
for m in range(1, 20, 5):
    if m == 1:
        plt.subplot(4, 5, m), plt.imshow(imglist[int(m / 5)][0]), plt.title('gt_61'), plt.ylabel('B')
        plt.axis('off')
        plt.subplot(4, 5, m + 1), plt.imshow(imglist[int(m / 5)][1]), plt.title('noise_61')
        plt.axis('off')
        plt.subplot(4, 5, m + 2), plt.imshow(imglist[int(m / 5)][2], 'gray'), plt.title('gt Fourier')
        plt.axis('off')
        plt.subplot(4, 5, m + 3), plt.imshow(imglist[int(m / 5)][3], 'gray'), plt.title('noise Fourier')
        plt.axis('off')
        plt.subplot(4, 5, m + 4), plt.imshow(imglist[int(m / 5)][4], 'gray'), plt.title('|gt - noise|')
        plt.axis('off')
    else:
        plt.subplot(4, 5, m), plt.imshow(imglist[int(m / 5)][0]), plt.ylabel('R' if m == 16 else 'G')
        plt.axis('off')
        plt.subplot(4, 5, m + 1), plt.imshow(imglist[int(m / 5)][1])
        plt.axis('off')
        plt.subplot(4, 5, m + 2), plt.imshow(imglist[int(m / 5)][2], 'gray')
        plt.axis('off')
        plt.subplot(4, 5, m + 3), plt.imshow(imglist[int(m / 5)][3], 'gray')
        plt.axis('off')
        plt.subplot(4, 5, m + 4), plt.imshow(imglist[int(m / 5)][4], 'gray')
        plt.axis('off')
plt.tight_layout()

plt.savefig('./fourier_61.tif', dpi=1000)

# plt.subplot(451), plt.imshow(img1, 'gray'), plt.title('gt_1')
# plt.axis('off')
# plt.subplot(452), plt.imshow(cimg1, 'gray'), plt.title('noise_1')
# plt.axis('off')
# plt.subplot(453), plt.imshow(fimg1, 'gray'), plt.title('gt Fourier')
# plt.axis('off')
# plt.subplot(454), plt.imshow(c_img1, 'gray'), plt.title('noise Fourier')
# plt.axis('off')
# plt.subplot(455), plt.imshow(newdel1, 'gray'), plt.title('|gt - noise|')
# plt.axis('off')
# plt.subplot(456), plt.imshow(img2, 'gray')
# plt.axis('off')
# plt.subplot(457), plt.imshow(cimg2, 'gray')
# plt.axis('off')
# plt.subplot(458), plt.imshow(fimg2, 'gray')
# plt.axis('off')
# plt.subplot(459), plt.imshow(c_img2, 'gray')
# plt.axis('off')
# plt.subplot(4, 5, 10), plt.imshow(newdel2, 'gray')
# plt.axis('off')
#
# plt.tight_layout()
#
# plt.savefig('./fourier_1_B.tif', dpi=1000)
