import os
from torch import nn
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# from result.algorithm.models.network import MIMOUNet
from utils import *
from MIMO.network import MIMOUNet
from demo_code.unetTorch import Unet

if __name__ == '__main__':
    t_path = r'../dataset/val/noisy/'
    gt = r'../dataset/train/gt/0_gt.dng'
    n_list = os.listdir(t_path)
    net = Unet()
    model_path = r'H:\中兴比赛\demo_code\baseline\models\th_model.pth'
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    net.eval()
    # io.imsave(output_n, result_write_data)
    for a, i in enumerate(n_list):
        # i = n_list[9]
        input_path = t_path + '/' + i
        num = i.split("_")
        output_path = r'H:\中兴比赛\dataset\val\\grad\\' + num[0] + '_grad' + i[-4:]
        raw_data_expand_c, height, width = read_image(input_path)

        raw_data_expand_c = cv2.medianBlur(raw_data_expand_c, (3, 3))

        raw_data_expand_c_normal = normalization(raw_data_expand_c, 1024, 16383)
        raw_data_expand_c_normal = torch.from_numpy(np.transpose(
            raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
        # raw_data_expand_c_normal = raw_data_expand_c_normal.cuda()
        # grad = img_gradient_total(raw_data_expand_c_normal)
        output = net(raw_data_expand_c_normal)

        grad = img_gradient_total(output.cuda())

        # grad = raw_data_expand_c_normal - grad

        result_data = grad.cpu().detach().numpy().transpose(0, 2, 3, 1)
        result_data = inv_normalization(result_data, 1024, 16383)
        result_write_data = write_image(result_data, height, width)
        print("result_write_data.shape:", result_write_data.shape)
        write_back_dng(gt, output_path, result_write_data)
