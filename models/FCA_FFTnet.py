"""
代码基于DeepRFT_MIMO修改

@inproceedings{,
    title={Deep Residual Fourier Transformation for Single Image Deblurring},
    author={Xintian Mao, Yiming Liu, Wei Shen, Qingli Li, Yan Wang},
    booktitle={arXiv:2111.11745},
    year={2021}
}
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch._jit_internal import Optional
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class DOConv2d(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size=3, D_mul=None, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', simam=False):
        super(DOConv2d, self).__init__()

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        self.simam = simam
        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.D_diag = Parameter(D_diag, requires_grad=False)
        ##################################################################################################
        if simam:
            self.simam_block = simam_module()
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.D_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            DoW = torch.reshape(self.W, DoW_shape)
        if self.simam:
            DoW_h1, DoW_h2 = torch.chunk(DoW, 2, dim=2)
            DoW = torch.cat([self.simam_block(DoW_h1), DoW_h2], dim=2)

        return self._conv_forward(input, DoW)


class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1, sig=False, relu_method=nn.ReLU):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
        if sig:
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                         groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_do, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


# 通道注意力
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicConv(channel, channel // reduction, 1, 1, relu=True)
        self.c2 = BasicConv(channel // reduction, channel, 1, 1, sig=True)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2


# 频域通道注意

class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward', att=False):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.att = att
        if self.att:
            c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7), (32, 128)])
            # self.at = CALayer(out_channel, reduction=8)
            self.att = MultiSpectralAttentionLayer(out_channel, c2wh[out_channel], c2wh[out_channel], reduction=16,
                                                   freq_sel_method='top16')
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.att:
            yy = self.main(x)
            ca = self.att(yy)
            out = ca + x + y
        else:
            out = self.main(x) + x + y
        return out


def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]


def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = torch.zeros([B, C, H, W], device=windows.device)
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock_do_fft_bench, att=False):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, att=att) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock_do_fft_bench, att=False):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, att=att) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class AFF_att_Fca(nn.Module):
    def __init__(self, in_channel, out_channel, att=True):
        super(AFF_att_Fca, self).__init__()
        self.att = att
        self.conv1 = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True, relu_method=nn.LeakyReLU),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        if self.att:
            c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7), (32, 128)])
            # self.att = CALayer(out_channel, reduction=8)
            self.att = MultiSpectralAttentionLayer(out_channel, c2wh[out_channel], c2wh[out_channel],
                                                   reduction=16, freq_sel_method='top16')
        # self.conv = BasicConv(out_channel * 2, out_channel * 2, kernel_size=1, stride=1,
        #                       relu=True, relu_method=nn.LeakyReLU)

    def forward(self, x, up):
        y1 = torch.cat([x, up], dim=1)
        y2 = self.conv1(y1)
        att = self.att(y2)
        return torch.cat([att + x, up], dim=1)


class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=4):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - inchannel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class Subspace(nn.Module):

    def __init__(self, in_size, out_size, relu_slope=0.2, subspace_dim=16):
        super(Subspace, self).__init__()
        self.size = in_size
        self.blocks = []
        self.blocks.append(nn.Conv2d(in_size, out_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.blocks.append(nn.LeakyReLU(relu_slope))
        self.blocks.append(nn.Conv2d(in_size, out_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.blocks.append(nn.LeakyReLU(relu_slope))

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=(1, 1), bias=True)

        self.main = nn.Sequential(*self.blocks)
        self.num_subspace = subspace_dim

    def forward(self, a, b):
        # b 上采样后  a 跳跃连接
        x = torch.cat([b, a], dim=1)
        sc = self.shortcut(x)

        sub = self.main(x) + sc  # 128

        b_, c_, h_, w_ = a.shape
        V_t = sub.reshape(b_, self.size, h_ * w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(dim=2, keepdim=True))
        V = V_t.permute(0, 2, 1)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.pinverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = a.reshape(b_, c_, h_ * w_)
        project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = torch.matmul(V, project_feature).permute(0, 2, 1).reshape(b_, c_, h_, w_)

        out = torch.cat([b, bridge], 1)
        return out


class DeepRFT(nn.Module):
    def __init__(self, num_res=8, inference=False):
        super(DeepRFT, self).__init__()
        self.inference = inference
        if not inference:
            BasicConv = BasicConv_do
            ResBlock = ResBlock_do_fft_bench

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock=ResBlock, att=False),
            EBlock(base_channel * 2, num_res, ResBlock=ResBlock, att=False),
            EBlock(base_channel * 4, num_res, ResBlock=ResBlock, att=False),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(4, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 4, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, ResBlock=ResBlock, att=False),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlock, att=False),
            DBlock(base_channel, num_res, ResBlock=ResBlock, att=False)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 4, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 4, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel * 2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

        self.AFF_fca1 = AFF_att_Fca(base_channel * 2, base_channel)

        self.AFF_fca2 = AFF_att_Fca(base_channel * 4, base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)  # 32 * 7 -> 32 * 2
        res1 = self.AFFs[0](res1, z21, z41)  # 32 * 7 -> 32 * 1

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)  # 上采样128->64

        if not self.inference:
            outputs.append(z_ + x_4)

        # z = torch.cat([z, res2], dim=1)  # 跳跃连接 64->128
        # z = self.SSA1(res2, z)  # ssa
        z = self.AFF_fca2(res2, z)

        z = self.Convs[0](z)  # 128 ->64
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)  # 最底层输出
        z = self.feat_extract[4](z)  # 上采样64->32
        if not self.inference:
            outputs.append(z_ + x_2)

        # z = torch.cat([z, res1], dim=1)  # 跳跃连接 32->64
        # z = self.SSA2(res1, z)  # ssa
        z = self.AFF_fca1(res1, z)

        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)  # 输出
        if not self.inference:
            outputs.append(z + x)
            # print(outputs)
            return outputs
        else:
            return z + x


class HalfInstanceNorm2d(nn.InstanceNorm2d):

    # MARK: Magic Functions

    def __init__(self, num_features: int, affine: bool = True, *args, **kwargs):
        super().__init__(
            num_features=num_features // 2, affine=affine, *args, **kwargs
        )

    # MARK: Forward Pass

    def forward(self, input):
        self._check_input_dim(input)
        out_1, out_2 = torch.chunk(input, 2, dim=1)
        out_1 = F.instance_norm(
            out_1, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum,
            self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


if __name__ == '__main__':
    model = DeepRFT(8)
    image1 = torch.randn(1, 4, 256, 8)
    start_time = time.time()
    with torch.no_grad():
        output1 = model.forward(image1)
    print(output1[2].shape)
    print('training time:', time.time() - start_time)
