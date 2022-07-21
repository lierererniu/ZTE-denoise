# ZTE-denoise
中兴捧月-算法赛道，初赛58.27分方案
链接：https://pan.quark.cn/s/7c2c6717387a  提取码：yhEs

ZTE-denoise

``` 
ZTE-denoise  
├─datanpy
		│  ├─train
		│  │  ├─gt
		│  │  │      gt.npy
		│  │  │      
		│  │  └─noisy
		│  │          noise.npy
		│  │          
		│  └─val
		│      ├─gt
		│      │      gt.npy
		│      └─noisy
		│              noise.npy
		│              
		├─dataset
		│  ├─test
		│  │      noisyx.dng
		│  │      
		│  ├─train
		│  │  ├─gt
		│  │  │      x_gt.dng
		│  │  │      
		│  │  └─noisy
		│  │          x_noise.dng
		│  │          
		│  └─val
		│      ├─gt
		│      │      61_gt.dng
		│      │      99_gt.dng
		│      │      
		│      └─noisy
		│              61_noise.dng
		│              99_noise.dng    
		├─models
		│  │  FCA_FFTnet.py   
		├─result
		│  │  ...
		│  ├─algorithm
		│  │  └─models
		│  │      │  ...
		│  └─data
		│          denoisex.dng 
		├─testset
		│      noisyx.dng
		│      
		├─utils
		│  │  ...

 ```

# 解题思路

## 赛题任务分解

1. RAW域去噪问题
2. 模型参数文件大小50M
3. 建议类Unet模型

## 去噪方法选择

- 基于滤波的方法
  - 训练集包含五种噪声模型，传统滤波方法泛化能力弱，面对五种噪声难以达到好的效果。
- 基于模型的方法
  - 涉及复杂的优化问题，去噪过程非常耗时，并且需要针对特定水平的噪音训练特定的模型，在盲图像去噪上受限。
- 基于学习的方法
  - 泛化能力强，更适合多噪声混合的任务。

## 数据处理方法

1. 随机裁剪

   耗时，收敛慢，随机性较大，需要很大的epoch才能保证训练样本覆盖全面。

2. 裁剪后保存为npy

   dng格式含有表头信息，裁切后表头信息变化，无法保存回dng格式，所以选择裁剪后保存为.npy文件

## 比较不同模型的效果

1. 类U-net模型

2. CBDnet  Toward Convolutional Blind Denoising of Real Photographs

   对于盲去噪任务，无法获得噪声分布，它的噪声估计子网络无法训练；

3. MIMO-net   Rethinking Coarse-to-Fine Approach in Single Image Deblurring

   去模糊的文章，与去噪同为low-level任务，它的方法在去噪任务中同样适用；

4. DeepRFT   Deep Residual Fourier Transformation for Single Image Deblurring

   MIMO-net的升级版，从图像的频域角度出发；

   | **模型** | **CBDnet(随机裁剪**) | MIMOnet(随机裁剪) | MIMOnet(全样本npy) | DeepRFT(全样本npy) |
   | :------: | -------------------- | ----------------- | ------------------ | ------------------ |
   |   分数   | 47.76                | 53.04             | 54.93              | 56.99              |

## 数据处理

### 验证集

选取61_xxx.dng和99_xxx.dng作为验证集，裁剪为256 * 256 * 4大小，裁剪后共生成108对数据。

### 训练集 

98对数据作为训练集，裁剪为256 * 256 * 4大小。

采用了4种数据增广方法，共产生了31752对数据：

•旋转：将输入随机旋转 90 度零次或多次。旋转概率p=0.5

•上下翻转：翻转概率p=0.5

•左右翻转：翻转概率p=0.5

•加噪：高斯噪声、椒盐噪声、乘性噪声、混合噪声(椒盐+高斯、泊松+高斯)

数据集裁剪和增广代码见 utils目录下 DataAugment.py    utils/DataAugment.py

## 模型

### 模型是基于DeepRFT修改的
![image2](https://github.com/lierererniu/picnote/blob/main/img/%E5%9B%BE%E7%89%872.png)
### FCAM

![图片4](https://github.com/lierererniu/picnote/blob/main/img/%E5%9B%BE%E7%89%874.jpg)

### 部分结果展示

![图片5](https://github.com/lierererniu/picnote/blob/main/img/%E5%9B%BE%E7%89%875.bmp)

| **模型** | **MIMOnet** | **DeepRFT** | **DeepRFAT** |
| :------: | :---------: | :---------: | :----------: |
|   分数   |    54.93    |    57.59    |    58.27     |

## 训练参数设置

- 网络权重初始化——采用torch默认初始化，未做特殊初始化
- Batchsize：6
- 每层Encoder和Decoder数量：8
- Epoch：200
- 学习率𝑙𝑟：0.0003
- 学习率衰减策略：阶梯衰减--每10个epoch衰减到原来的80%
- 优化器采用Adam，其中betas=(0.9, 0.999)，eps=1e-8，weight_decay=0
