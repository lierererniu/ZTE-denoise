import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# swim
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return


# 2维图像转为1维 Patch Embeddings
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches
        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)  # 归一化
        else:
            self.norm = None

    # 定义前向传播
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x


# 从 1维变2维 Embeddings 组合图像
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding
    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches
        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # 输出结构为 [B, Ph*Pw, C]
        return x


# 将输入分割为多个不重叠窗口
def window_partition(x, window_size):
    """
    输入:
        x: (B, H, W, C)
        window_size (int): window size  # 窗口的大小
    返回:
        windows: (num_windows*B, window_size, window_size, C)  # 每一个 batch 有单独的 windows
    """
    B, H, W, C = x.shape  # 输入的 batch 个数，高，宽，通道数
    # 将输入 x 重构为结构 [batch 个数，高方向的窗口个数，窗口大小，宽方向的窗口个数，窗口大小，通道数] 的张量
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 交换重构后 x 的第 3和4 维度， 5和6 维度，再次重构为结构 [高和宽方向的窗口个数乘以 batch 个数，窗口大小，窗口大小，通道数] 的张量
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
    # 这里比较有意思，不太理解的可以给个初始值，比如 x = torch.randn([1, 14, 28, 3])


# 窗口注意力
class WindowAttention(nn.Module):
    r""" 基于有相对位置偏差的多头自注意力窗口，支持移位的(shifted)或者不移位的(non-shifted)窗口.
    输入:
        dim (int): 输入特征的维度.
        window_size (tuple[int]): 窗口的大小.
        num_heads (int): 注意力头的个数.
        qkv_bias (bool, optional): 给 query, key, value 添加可学习的偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        attn_drop (float, optional): 注意力权重的丢弃率，默认为 0.0.
        proj_drop (float, optional): 输出的丢弃率，默认为 0.0.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.window_size = window_size  # 窗口的高 Wh,宽 Ww
        self.num_heads = num_heads  # 注意力头的个数
        head_dim = dim // num_heads  # 注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子 scale
        # 定义相对位置偏移的参数表，结构为 [2*Wh-1 * 2*Ww-1, num_heads]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        # 获取窗口内每个 token 的成对的相对位置索引
        coords_h = torch.arange(self.window_size[0])  # 高维度上的坐标 (0, 7)
        coords_w = torch.arange(self.window_size[1])  # 宽维度上的坐标 (0, 7)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 坐标，结构为 [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # 重构张量结构为 [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 相对坐标，结构为 [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 交换维度，结构为 [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 第1个维度移位
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 第1个维度移位
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 第1个维度的值乘以 2倍的 Ww，再减 1
        relative_position_index = relative_coords.sum(-1)  # 相对位置索引，结构为 [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)  # 保存数据，不再更新
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性层，特征维度变为原来的 3倍
        self.attn_drop = nn.Dropout(attn_drop)  # 随机丢弃神经元，丢弃率默认为 0.0
        self.proj = nn.Linear(dim, dim)  # 线性层，特征维度不变
        self.proj_drop = nn.Dropout(proj_drop)  # 随机丢弃神经元，丢弃率默认为 0.0
        trunc_normal_(self.relative_position_bias_table, std=.02)  # 截断正态分布，限制标准差为 0.02
        self.softmax = nn.Softmax(dim=-1)  # 激活函数 softmax

    # 定义前向传播
    def forward(self, x, mask=None):
        """
        输入:
            x: 输入特征图，结构为 [num_windows*B, N, C]
            mask: (0/-inf) mask, 结构为 [num_windows, Wh*Ww, Wh*Ww] 或者没有 mask
        """
        B_, N, C = x.shape  # 输入特征图的结构
        # 将特征图的通道维度按照注意力头的个数重新划分，并再做交换维度操作
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 方便后续写代码，重新赋值
        # q 乘以缩放因子
        q = q * self.scale
        # @ 代表常规意义上的矩阵相乘
        attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
        # 相对位置偏移，结构为 [Wh*Ww, Wh*Ww, num_heads]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # 相对位置偏移交换维度，结构为 [num_heads, Wh*Ww, Wh*Ww]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)  # 带相对位置偏移的注意力图
        if mask is not None:  # 判断是否有 mask
            nW = mask.shape[0]  # mask 的宽
            # 注意力图与 mask 相加
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)  # 恢复注意力图原来的结构
            attn = self.softmax(attn)  # 激活注意力图 [0, 1] 之间
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)  # 随机设置注意力图中的部分值为 0
        # 注意力图与 v 相乘得到新的注意力图
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)  # 通过线性层
        x = self.proj_drop(x)  # 随机设置新注意力图中的部分值为 0
        return x


# 将多个不重叠窗口重新合并
def window_reverse(windows, window_size, H, W):
    """
    输入:
        windows: (num_windows*B, window_size, window_size, C)  # 分割得到的窗口(已处理)
        window_size (int): Window size  # 窗口大小
        H (int): Height of image  # 原分割窗口前特征图的高
        W (int): Width of image  # 原分割窗口前特征图的宽
    返回:
        x: (B, H, W, C)  # 返回与分割前特征图结构一样的结果
    """
    # 以下就是分割窗口的逆向操作，不多解释
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# Swin Transformer 块
class SwinTransformerBlock(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入特征图的分辨率.
        num_heads (int): 注意力头的个数.
        window_size (int): 窗口的大小.
        shift_size (int): SW-MSA 的移位值.
        mlp_ratio (float): 多层感知机隐藏层的维度和嵌入层的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): 随机神经元丢弃率，默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float, optional): 深度随机丢弃率，默认为 0.0.
        act_layer (nn.Module, optional): 激活函数，默认为 nn.GELU.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.input_resolution = input_resolution  # 输入特征图的分辨率
        self.num_heads = num_heads  # 注意力头的个数
        self.window_size = window_size  # 窗口的大小
        self.shift_size = shift_size  # SW-MSA 的移位大小
        self.mlp_ratio = mlp_ratio  # 多层感知机隐藏层的维度和嵌入层的比
        if min(self.input_resolution) <= self.window_size:  # 如果输入分辨率小于等于窗口大小
            self.shift_size = 0  # 移位大小为 0
            self.window_size = min(self.input_resolution)  # 窗口大小等于输入分辨率大小
        # 断言移位值必须小于等于窗口的大小
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)  # 归一化层
        # 窗口注意力
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # 如果丢弃率大于 0 则进行随机丢弃，否则进行占位(不做任何操作)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # 归一化层
        mlp_hidden_dim = int(dim * mlp_ratio)  # 多层感知机隐藏层维度
        # 多层感知机
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:  # 如果移位值大于 0
            attn_mask = self.calculate_mask(self.input_resolution)  # 计算注意力 mask
        else:
            attn_mask = None  # 注意力 mask 赋空
        self.register_buffer("attn_mask", attn_mask)  # 保存注意力 mask，不参与更新

    # 计算注意力 mask
    def calculate_mask(self, x_size):
        H, W = x_size  # 特征图的高宽
        img_mask = torch.zeros((1, H, W, 1))  # 新建张量，结构为 [1, H, W, 1]
        # 以下两 slices 中的数据是索引，具体缘由尚未搞懂
        h_slices = (slice(0, -self.window_size),  # 索引 0 到索引倒数第 window_size
                    slice(-self.window_size, -self.shift_size),  # 索引倒数第 window_size 到索引倒数第 shift_size
                    slice(-self.shift_size, None))  # 索引倒数第 shift_size 后所有索引
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt  # 将 img_mask 中 h, w 对应索引范围的值置为 cnt
                cnt += 1  # 加 1
        mask_windows = window_partition(img_mask, self.window_size)  # 窗口分割，返回值结构为 [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)  # 重构结构为二维张量，列数为 [window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 增加第 2 维度减去增加第 3 维度的注意力 mask
        # 用浮点数 -100. 填充注意力 mask 中值不为 0 的元素，再用浮点数 0. 填充注意力 mask 中值为 0 的元素
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    # 定义前向传播
    def forward(self, x, x_size):
        H, W = x_size  # 输入特征图的分辨率
        B, L, C = x.shape  # 输入特征的 batch 个数，长度和维度
        # assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)  # 归一化
        x = x.view(B, H, W, C)  # 重构 x 为结构 [B, H, W, C]
        # 循环移位
        if self.shift_size > 0:  # 如果移位值大于 0
            # 第 0 维度上移 shift_size 位，第 1 维度左移 shift_size 位
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x  # 不移位
        # 对移位操作得到的特征图分割窗口, nW 是窗口的个数
        x_windows = window_partition(shifted_x, self.window_size)  # 结构为 [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # 结构为 [nW*B, window_size*window_size, C]
        # W-MSA/SW-MSA, 用在分辨率是窗口大小的整数倍的图像上进行测试
        if self.input_resolution == x_size:  # 输入分辨率与设定一致，不需要重新计算注意力 mask
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # 注意力窗口，结构为 [nW*B, window_size*window_size, C]
        else:  # 输入分辨率与设定不一致，需要重新计算注意力 mask
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,
                                         C)  # 结构为 [-1, window_size, window_size, C]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # 结构为 [B, H', W', C]
        # 逆向循环移位
        if self.shift_size > 0:
            # 第 0 维度下移 shift_size 位，第 1 维度右移 shift_size 位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x  # 不逆向移位
        x = x.view(B, H * W, C)  # 结构为 [B, H*W， C]
        # FFN
        x = shortcut + self.drop_path(x)  # 对 x 做 dropout，引入残差
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 归一化后通过 MLP，再做 dropout，引入残差
        return x


# 单阶段的 SWin Transformer 基础层
class BasicLayer(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入分辨率.
        depth (int): SWin Transformer 块的个数.
        num_heads (int): 注意力头的个数.
        window_size (int): 本地(当前块中)窗口的大小.
        mlp_ratio (float): MLP隐藏层特征维度与嵌入层特征维度的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): 随机丢弃神经元，丢弃率默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float | tuple[float], optional): 深度随机丢弃率，默认为 0.0.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
        downsample (nn.Module | None, optional): 结尾处的下采样层，默认没有.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.input_resolution = input_resolution  # 输入分辨率
        self.depth = depth  # SWin Transformer 块的个数
        # 创建 Swin Transformer 网络
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        # patch 合并层
        if downsample is not None:  # 如果有下采样
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)  # 下采样
        else:
            self.downsample = None  # 不做下采样

    # 定义前向传播
    def forward(self, x, x_size):
        for blk in self.blocks:  # x 输入串联的 Swin Transformer 块
            x = blk(x, x_size)  # 直接输入网络

        return x


# if __name__ == '__main__':
#     model = BasicLayer(dim=4,
#                        input_resolution=(1, 1),
#                        depth=[6, 6, 6, 6],
#                        num_heads=[6, 6, 6, 6],
#                        window_size=7,
#                        mlp_ratio=4.0,
#                        qkv_bias=True, qk_scale=0.,
#                        drop=0., attn_drop=0.,
#                        drop_path=0.,
#                        norm_layer=nn.LayerNorm)
