import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio
from torch.autograd import Variable
from utils.utils import inv_normalization, write_image
import warnings
warnings.filterwarnings("ignore")

def validation(config, test_data_loader, model, criterionMSE, epoch):
    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    model.eval()

    for i, data in enumerate(test_data_loader):
        gt, noise = Variable(data[0]), Variable(data[1])

        if config.cuda:
            gt = gt.cuda()
            noise = noise.cuda()
        with torch.no_grad():
            fake = model(noise)

        fake_ = fake[2].cpu().detach().numpy().transpose(0, 2, 3, 1)
        result = inv_normalization(fake_, config.black_level, config.white_level)
        result_ = write_image(result, 512, 512)
        gt_ = gt.cpu().detach().numpy().transpose(0, 2, 3, 1)
        gt_ = inv_normalization(gt_, config.black_level, config.white_level)
        gt_ = write_image(gt_, 512, 512)

        mse = criterionMSE(fake[2], gt)
        # psnr = 10 * np.log10(1 / mse.item())
        psnr = peak_signal_noise_ratio(gt_.astype(np.float), result_.astype(np.float),
                                       data_range=config.white_level)
        ssim = SSIM(gt_.astype(np.float), result_.astype(np.float),
                    multichannel=True, data_range=config.white_level)
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim

    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    print("===> Avg. MSE: {:.4f}".format(np.sqrt(avg_mse)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f}".format(avg_ssim))

    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim
    return log_test, avg_psnr, avg_ssim
