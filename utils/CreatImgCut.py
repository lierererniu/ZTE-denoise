#  ###write by WangJinShuai
import os
import numpy as np
import rawpy
import torch
import skimage.metrics
from matplotlib import pyplot as plt
import argparse
import glob
from models.FCA_FFTnet import DeepRFT


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
    # print(raw_data.shape)#(3472,4624)
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


# def denoise_raw(input_path, output_path, ground_path, model_path, black_level, white_level):
def denoise_raw(input_path, output_path, model_path, black_level, white_level):
    """
    Example: obtain ground truth
    """
    # gt = rawpy.imread(ground_path).raw_image_visible

    """
    pre-process
    """
    raw_data_expand_c, height, width = read_image(input_path)
    m_h = 7  # each=185, r_h=71
    m_w = 10  # h=250, r_h=6
    ps_temp = 256
    new = np.zeros((1, 1736, 2312, 4), dtype=np.uint32)

    for i in range(0, m_h):
        y = i * 252
        for j in range(0, m_w):
            x = j * 252

            image = raw_data_expand_c[y:y + ps_temp, x:x + ps_temp, :]

            raw_data_expand_c_normal = normalization(image, black_level, white_level)
            h = raw_data_expand_c_normal.shape[0]
            w = raw_data_expand_c_normal.shape[1]
            # print(raw_data_expand_c_normal.shape)
            raw_data_expand_c_normal1 = torch.from_numpy(np.transpose(
                raw_data_expand_c_normal.reshape(-1, h, w, 4), (0, 3, 1, 2))).float()
            raw_data_expand_c_normal1 = raw_data_expand_c_normal1.cpu()
            net = DeepRFT(8).cpu()
            device = torch.device('cpu')
            if model_path is not None:
                net.load_state_dict(torch.load(model_path, map_location=device), False)
            net.eval()
            with torch.no_grad():
                # print(raw_data_expand_c_normal1.shape)
                result_data = net(raw_data_expand_c_normal1)
            # result_data = net(raw_data_expand_c_normal1)
            result_data = result_data[2].cpu().detach().numpy().transpose(0, 2, 3, 1)
            # print(result_data.shape)
            result_data = inv_normalization(result_data, black_level, white_level)  # [1,256,256,4]
            # print(result_data.shape)
            new[:, y:y + ps_temp, x:x + ps_temp, :] = new[:, y:y + ps_temp, x:x + ps_temp, :] + result_data

    for i in range(1, m_h):
        y = i * 252
        new[:, y:y + 4, :, :] = new[:, y:y + 4, :, :] / 2
    for j in range(1, m_w):
        x = j * 252
        new[:, :, x:x + 4, :] = new[:, :, x:x + 4, :] / 2

    result_write_data = write_image(new, height, width)
    write_back_dng(input_path, output_path, result_write_data)


def main(args):
    # load data
    name = os.listdir(args.input_path)
    print(name)
    for i in name:
        input_path = args.input_path + i
        print(input_path)
        # print('input_path',input_path)
        output_path = args.output_path + 'denoise' + i.split('y')[1]
        # print('output_path',output_path)

        model_path = args.model_path
        black_level = args.black_level
        white_level = args.white_level
        # input_path = args.input_path
        # output_path = args.output_path
        # ground_path = args.ground_path
        # denoise_raw(input_path, output_path, ground_path, model_path, black_level, white_level)
        denoise_raw(input_path, output_path, model_path, black_level, white_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=r"C:\Users\53110\Desktop\best.pth")
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    parser.add_argument('--input_path', type=str, default="../dataset/test/")
    parser.add_argument('--output_path', type=str, default=r"H:\result\result\data\\")

    args = parser.parse_args()
    main(args)
