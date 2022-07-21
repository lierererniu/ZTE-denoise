import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from result.algorithm.models.network import ResNet
from result.algorithm.models.CBDnet import Network
from utils.utils import *
from skimage import io
def predict(noise, path, use_cuda):
    # model = Spatio_temporal(120, 60, (1, 1), 1, (3, 3))
    device = torch.device('cpu')
    model = Network()
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    if use_cuda:
        noise = noise.cuda()
    with torch.no_grad():
        noise = torch.autograd.Variable(noise)
    outputs = model(noise)
    return outputs


if __name__ == '__main__':
    t_path = r'./dataset/test'

    n_list = os.listdir(t_path)
    modelpath = r'H:\中兴比赛\result\algorithm\models\model_epoch_0.pth'
    # test1 = r'H:\中兴比赛\data\train\noisy\0_noise.tif'
    # gt1 = r'H:\中兴比赛\data\train\gt\0_gt.tif'
    # gt = gdal.Open(gt1).ReadAsArray()
    # raw_data_expand_c, height, width = read_image(test1)
    # raw_data_expand_c_normal = normalization(raw_data_expand_c, 1024, 16383)
    # raw_data_expand_c_normal = torch.from_numpy(np.transpose(
    #     raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
    # output = predict(raw_data_expand_c_normal, modelpath, False)
    # result_data = output.cpu().detach().numpy().transpose(0, 2, 3, 1)
    # result_data = inv_normalization(result_data, 1024, 16383)
    # result_write_data = write_image(result_data, height, width).astype(np.uint16).tobytes()
    # output_n = r'H:\中兴比赛\denoise.tif'
    # # cv2.imwrite(output_n, result_write_data)
    # io.imsave(output_n, result_write_data)
    for i in n_list:
        input_path = t_path + '/' + i
        output_path = r'./result/data/' + 'denoise' + i[-5:]
        raw_data_expand_c, height, width = read_image(input_path)
        raw_data_expand_c_normal = normalization(raw_data_expand_c, 1024, 16383)
        raw_data_expand_c_normal = torch.from_numpy(np.transpose(
            raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
        _, output = predict(raw_data_expand_c_normal, modelpath, False)
        result_data = output.detach().numpy().transpose(0, 2, 3, 1)
        result_data = inv_normalization(result_data, 1024, 16383)
        result_write_data = write_image(result_data, height, width)
        print("result_write_data.shape:", result_write_data.shape)
        write_back_dng(input_path, output_path, result_write_data)


