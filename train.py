import argparse
import os
import time
import numpy as np
import torch
import albumentations as A
from torch.utils.data import DataLoader
from utils.data_torch import TrainDataset, ValDataset
from utils.lossReport import lossReport, TestReport
from torch import nn
from utils.utils import gpu_manage, checkpoint, rfft
from validation import validation
import torch.nn.functional as F
from focal_frequency_loss import FocalFrequencyLoss as FFL
ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class
from models.FCA_FFTnet import DeepRFT


trfm = A.Compose([
    A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.RandomCrop(256, 256, always_apply=True, p=1)
])

Sl1loss = nn.SmoothL1Loss()
# criterion = fixed_loss()
criterion = nn.MSELoss()


def adjust_learning_rate(optimizer, epoch, lr, step):
    """
    动态lr 每20次epoch调整一次
    :param optimizer: 优化器
    :param epoch: 迭代次数
    :param lr: 学习率
    :return: None
    """
    lr = lr * (0.8 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):
    rem = 0
    w = 0
    gpu_manage(args)
    dataset = TrainDataset(args, trfm)
    vdataset = ValDataset(args)
    print('train dataset:', len(dataset))
    print('validation dataset:', len(vdataset))
    training_data_loader = DataLoader(dataset=dataset, num_workers=args.threads, batch_size=args.batchsize,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=vdataset, num_workers=args.threads,
                                        batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepRFT(8).type(torch.float32)
    # 师傅加载预训练
    path = r'./result/algorithm/models/best.pth'
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

    criterionMSE = nn.MSELoss()
    l1loss = nn.L1Loss()
    if args.cuda:
        model = model.cuda()
        criterionMSE = criterionMSE.cuda()
        criterion.cuda()
        l1loss.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k == 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)
    logreport = lossReport(log_dir=args.output_path)
    validationreport = TestReport(log_dir=args.output_path)
    print('===> begin')
    start_time = time.time()
    best_ssim = -1
    best_psnr = -1
    for epoch in range(args.epoch):
        model.train()
        epoch_start_time = time.time()
        adjust_learning_rate(optimizer, epoch, args.lr, 1)
        losses = np.ones((len(training_data_loader)))
        for batch_idx, data in enumerate(training_data_loader):
            gt, nosie = data[0].cuda(), data[1].cuda()
            pred_img = model(nosie)
            label_img2 = F.interpolate(gt, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(gt, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], gt)
            loss_content = l1 + l2 + l3

            label_fft1 = rfft(label_img4)
            pred_fft1 = rfft(pred_img[0])
            label_fft2 = rfft(label_img2)
            pred_fft2 = rfft(pred_img[1])
            label_fft3 = rfft(gt)
            pred_fft3 = rfft(pred_img[2])
            ff1 = l1loss(pred_fft1, label_fft1)
            ff2 = l1loss(pred_fft2, label_fft2)
            ff3 = l1loss(pred_fft3, label_fft3)
            loss_fftf = ff1 + ff2 + ff3
            f1 = l1loss(pred_img[0], label_img4)
            f2 = l1loss(pred_img[1], label_img2)
            f3 = l1loss(pred_img[2], gt)
            loss_fft = f1 + f2 + f3
            loss = loss_content + 0.8 * loss_fft + 0.2 * loss_fftf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[batch_idx] = loss.item()
            print(
                "===> Epoch[{}]({}/{}): loss: {:.4f} ".format(
                    epoch, batch_idx, len(training_data_loader), loss.item()))
        log = {}
        log['epoch'] = epoch
        log['loss'] = np.average(losses)
        logreport(log)
        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation, psnr, ssim = validation(args, validation_data_loader, model, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        rem = rem + psnr
        w = w + ssim
        if epoch % args.snapshot_interval == 0 or epoch == args.epoch - 1:
            checkpoint(args, epoch, model)
        if epoch % 1 == 0:
            rem = rem
            w = w
            if rem >= best_psnr and w >= best_ssim:
                print("epoch{}, max_psnr: {:.4f} dB.".format(epoch, rem))
                torch.save(model.state_dict(), os.path.join(args.output_path + r'/algorithm/models', 'best.pth'))
                best_psnr = rem
                best_ssim = w
            if rem > best_psnr:
                print("epoch{}, max_psnr: {:.4f} dB.".format(epoch, rem))
                torch.save(model.state_dict(), os.path.join(args.output_path + r'/algorithm/models', 'best_psnr.pth'))
                best_psnr = rem
            if w > best_ssim:
                print("epoch{}, max_ssim: {:.4f}.".format(epoch, w))
                torch.save(model.state_dict(), os.path.join(args.output_path + r'/algorithm/models', 'best_ssim.pth'))
                best_ssim = w
            rem = 0
            w = 0
        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    parser.add_argument('--datasets_dir', type=str, default="./datanpy/train/")
    parser.add_argument('--valsets_dir', type=str, default="./datanpy/val/")
    parser.add_argument('--input_path', type=str, default="noisy")
    parser.add_argument('--output_path', type=str, default="./result")
    parser.add_argument('--ground_path', type=str, default="gt")
    parser.add_argument('--grad_path', type=str, default="grad")
    parser.add_argument('--train_list', type=str, default="train_list.txt")
    parser.add_argument('--val_list', type=str, default="val_list.txt")
    parser.add_argument('--train_size', type=int, default="1")
    parser.add_argument('--validation_size', type=float, default="0.1")
    parser.add_argument('--batchsize', type=int, default="6")
    parser.add_argument('--cuda', type=bool, default="true")
    parser.add_argument('--gpu_ids', type=str, default="0")
    parser.add_argument('--manualSeed', type=int, default="0")
    parser.add_argument('--threads', type=int, default="4")
    parser.add_argument('--lr', type=float, default="0.00004")
    parser.add_argument('--epoch', type=int, default="200")
    parser.add_argument('--snapshot_interval', type=int, default="5")
    args = parser.parse_args()
    train(args)
