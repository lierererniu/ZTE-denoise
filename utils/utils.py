import os

import skimage
import torch
import random

from osgeo import gdal
from torch import nn
from torch.backends import cudnn
import numpy as np
import rawpy


def checkpoint(args, epoch, model):
    model_dir = args.output_path + r'/algorithm/models'
    # model_dir = os.path.join(args.output_path, r'\algorithm\models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_out_path = os.path.join(model_dir, 'model_epoch_{}.pth'.format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_dir))


def gpu_manage(args):
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))
        args.gpu_ids = list(range(len(args.gpu_ids)))

    print(os.environ['CUDA_VISIBLE_DEVICES'])

    if args.manualSeed == 0:
        args.manualSeed = random.randint(1, 10000)
    print('Random Seed: ', args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    # raw_data = gdal.Open(input_path).ReadAsArray()
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, height, width


def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
    return output_data


def rfft(image):
    label_fft1 = torch.fft.fft2(image, dim=(-2, -1))
    output = torch.stack((label_fft1.real, label_fft1.imag), -1)
    return output


def img_gradient_total(img):
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False, groups=4)
    a = torch.from_numpy(a).float().unsqueeze(0)
    a = torch.stack((a, a, a, a))
    conv1.weight = nn.Parameter(a, requires_grad=False)
    conv1 = conv1.cuda()
    G_x = conv1(img)

    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False, groups=4)
    b = torch.from_numpy(b).float().unsqueeze(0)
    b = torch.stack((b, b, b, b))
    conv2.weight = nn.Parameter(b, requires_grad=False)
    conv2 = conv2.cuda()
    G_y = conv2(img)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return G


def img_gradient(img):
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    a = torch.from_numpy(a).float().unsqueeze(0)
    a = torch.stack((a, a, a))
    conv1.weight = nn.Parameter(a, requires_grad=False)
    conv1 = conv1.cuda()
    G_x = conv1(img)

    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    b = torch.from_numpy(b).float().unsqueeze(0)
    b = torch.stack((b, b, b))
    conv2.weight = nn.Parameter(b, requires_grad=False)
    conv2 = conv2.cuda()
    G_y = conv2(img)

    return G_x, G_y


# 高斯噪声
def addGaussNoise(img):
    var = random.uniform(0.0001, 0.001)
    noisy = skimage.util.random_noise(img, mode='gaussian', var=var)
    return noisy


# 椒盐噪声
def addSaltNoise(img):
    var = random.uniform(0.01, 0.09)

    noisy = skimage.util.random_noise(img, mode='s&p', amount=var)
    return noisy


# 乘法噪声
def addSpeckleNoise(img):
    var = random.uniform(0.0001, 0.04)
    noisy = skimage.util.random_noise(img, mode='speckle', var=var)
    return noisy

# 泊松噪声
def addPoissonNoise(img):
    noisy = skimage.util.random_noise(img, mode='poisson')
    return noisy

#
def addNoise(data, mode):
    a = normalization(data, 1024, 16383)
    img = np.zeros((1, 256, 256, 4))
    if mode == 'gaussian':
        img = addGaussNoise(a).reshape(1, 256, 256, 4)
    if mode == 'speckle':
        img = addSpeckleNoise(a).reshape(1, 1736, 2312, 4)
    if mode == 's&p':
        img = addSaltNoise(a).reshape(1, 1736, 2312, 4)
    if mode == 'double_salt_gauss':
        img = addSaltNoise(a)
        img = addGaussNoise(img).reshape(1, 1736, 2312, 4)
    if mode == 'double_poisson_gauss':
        img = addSaltNoise(a)
        img = addPoissonNoise(img).reshape(1, 1736, 2312, 4)

    img = inv_normalization(img, 1024, 16383)
    return img.reshape(256, 256, 4)


def imgaug(gt, noise, height, width, item_height, item_width, num, gt_dic, noise_dic):
    for i in range(0, int((height // 2) / item_height)):
        for j in range(0, int((width // 2) / item_width)):
            crop_gt = gt[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
            # crop_noise = noise[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
            crop_noise = addNoise(crop_gt, mode='gaussian')
            gt_dic[num] = crop_gt
            noise_dic[num] = crop_noise
            num += 1
            print(num)
    return gt_dic, noise_dic, num
