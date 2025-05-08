

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
from torch.utils import data
import torch.multiprocessing
import random as rn
import numpy as np
import torch, os, pdb
from typing import Tuple

def loadTxt(fn):
    a = []
    with open(fn, 'r',encoding='gbk') as fp:
        data = fp.readlines()
        for item in data:
            fn = item.strip('\n')
            a.append(fn)
    return a


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - test/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('test' or 'val')
    """

    def __init__(self, root, transform=None, split="test"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        #img = Image.open(self.samples[index]).convert("RGB")
        # 将图像转换为灰度图像
        img = Image.open(self.samples[index]).convert("L")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, X, crop_size=128, width=128, root='', mode='Train', test_crop_size=256,test_width_size=256,num_channels=None):

        super(dataset_h5, self).__init__()

        self.root = root
        self.fns = X
        self.n_images = len(self.fns)
        self.indices = np.array(range(self.n_images))

        self.mode = mode
        self.crop_size = crop_size
        self.width = width
        self.test_crop_size = test_crop_size
        self.test_width_size = test_width_size
        self.num_channels = num_channels  # 新增通道数参数

    def __getitem__(self, index):
        data, label = {}, {}

        fn = os.path.join(self.root, self.fns[index])

        x = loadmat(fn)
        x = x[list(x.keys())[-1]]

        x = x.astype(np.float32)
        #img_size_h, img_size_w = x.shape[:2]
        img_size_h, img_size_w, num_total_channels = x.shape
        #         x[x<0]=0
        # 确保指定的通道数量不超过原图像的总通道数
        if self.num_channels is not None and self.num_channels <= num_total_channels:
            start_channel = rn.randint(0, num_total_channels - self.num_channels)
            x = x[:, :, start_channel:start_channel + self.num_channels]
        else:
            print(f"指定的通道数量 ({self.num_channels}) 超出可用范围，将使用所有通道。")


        if self.mode == 'Train':
            # Random crop
            shifting_h = (img_size_h - self.crop_size) // 2  # 计算垂直方向上可用于随机裁剪的范围
            shifting_w = (img_size_w - self.width) // 2  #
            xim, yim = rn.randint(0, shifting_w), rn.randint(0, shifting_h)
            h = yim + self.crop_size
            # xx = []

            y = x[yim:h, xim:xim + self.width, :]
            # Random flip
            if rn.random() > 0.5:
                y = y[::-1, :, :]
            if rn.random() > 0.5:
                y = y[:, ::-1, :]
            xmin = np.min(y)
            xmax = np.max(y)
            y = torch.from_numpy(y.copy())

            # xx.append(y)
            # x = torch.stack(y)
        else:
            # xx = []

            shifting_h = (img_size_h - self.test_crop_size) // 2  # 计算垂直方向上可用于随机裁剪的范围
            shifting_w = (img_size_w - self.test_width_size) // 2  #
            xim, yim = rn.randint(0, shifting_w), rn.randint(0, shifting_h)
            h = yim + self.test_crop_size
            # xx = []

            y = x[yim:h, xim:xim + self.test_width_size, :]
            xmin = np.min(y)
            xmax = np.max(y)
            y = torch.from_numpy(y.copy())

            # xx.append(y)
            # x = torch.stack(y)

        if xmin == xmax:
            print('nan in', self.fns[index])
            return np.zeros((1, 128, 4, 172))

        y = (y - xmin) / (xmax - xmin)  # 归一化
        # y=y/65535  #归一化
        return y, fn, xmax, xmin

    def __len__(self):
        return self.n_images


def split_sequence(sequence: np.ndarray, ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a sequence into src, tgt, and tgt_y as required by the transformer model."""
    src_end = int(sequence.shape[1] * ratio)
    src = sequence[:, :src_end]  # First part of the sequence
    tgt = sequence[:, src_end - 1 : -1]  # Shifted tgt
    tgt_y = sequence[:, src_end:]  # Second part of the sequence

    return src, tgt, tgt_y

class PixelSequenceDataset(Dataset):
    def __init__(self, image_paths, mode='Train', crop_size=128, width=128, num_channels=None, root='',sequence_length=5):
        super(PixelSequenceDataset, self).__init__()
        self.image_paths = image_paths
        self.mode = mode
        self.crop_size = crop_size
        self.width = width
        self.num_channels = num_channels
        self.root = root
        self.sequence_length = sequence_length



    def __len__(self):
        return  len(self.image_paths)  # 每张图像作为一个样本

    def __getitem__(self, idx):
        # 加载图像
        fn = os.path.join(self.root, self.image_paths[idx])
        x = loadmat(fn)
        x = x[list(x.keys())[-1]]

        x = x.astype(np.float32)
        img_size_h, img_size_w = x.shape[:2]
        num_total_channels = x.shape[2] if x.ndim == 3 else 1

        # 确保指定的通道数量不超过原图像的总通道数
        if self.num_channels is not None and self.num_channels <= num_total_channels:
            start_channel = rn.randint(0, num_total_channels - self.num_channels)
            x = x[:, :, start_channel:start_channel + self.num_channels]
        else:
            print(f"指定的通道数量 ({self.num_channels}) 超出可用范围，将使用所有通道。")
            self.num_channels = num_total_channels  # 更新通道数

        # 数据增强和裁剪
        if self.mode == 'Train':
            # 随机裁剪
            shifting_h = img_size_h - self.crop_size
            shifting_w = img_size_w - self.width
            xim = rn.randint(0, shifting_w) if shifting_w > 0 else 0
            yim = rn.randint(0, shifting_h) if shifting_h > 0 else 0
            y = x[yim:yim + self.crop_size, xim:xim + self.width, :]
            # 随机翻转
            if rn.random() > 0.5:
                y = y[::-1, :, :]
            if rn.random() > 0.5:
                y = y[:, ::-1, :]
        else:
            # 中心裁剪
            xim = (img_size_w - self.width) // 2 if img_size_w > self.width else 0
            yim = (img_size_h - self.crop_size) // 2 if img_size_h > self.crop_size else 0
            y = x[yim:yim + self.crop_size, xim:xim + self.width, :]

        # 归一化
        xmin = np.min(y)
        xmax = np.max(y)
        if xmin == xmax:
            print('图像中所有像素值相同，文件名：', self.image_paths[idx])
            y = np.zeros((self.crop_size * self.width, self.num_channels))
        else:
            y = (y - xmin) / (xmax - xmin)  # 归一化到 [0, 1]
            # 展平为1维序列
            #y_flat = y.flatten()  # 形状为 (128 * 128,)
            y_flat = y.reshape(-1)  # 形状为 (128 * 128, num_channels)
        # 生成移位序列
        sequences = []
        # for i in range(len(y_flat) - self.sequence_length):
        #     sequences.append(y_flat[i:i + self.sequence_length + 1])  # 包含当前像素用于预测
        for i in range(0, len(y_flat) - self.sequence_length, self.sequence_length):
            sequences.append(y_flat[i:i + self.sequence_length + 1])

        sequences = np.array(sequences)  # 转换为数组

        # 转换为张量并分离输入和目标
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)

        # src = sequences_tensor[:, :-1]  # 输入序列（过去N个像素）# 输入序列（过去N-1个像素
        # tgt = sequences_tensor[:, 1:]

        src, tgt, tgt_y = split_sequence(sequences_tensor)
        # src = sequences_tensor[:, :-1]  # 输入序列（前 sequence_length 个元素）
        # tgt = sequences_tensor[:, 1:-1]  # 解码器的输入（中间部分）
        # tgt_y = sequences_tensor[:, 1:]  # 真实目标序列（包含最后一个元素）
        # 目标序列（预测的像素）  形状是 (N, 5)
        return src, tgt ,tgt_y # 返回输入和目标序列


