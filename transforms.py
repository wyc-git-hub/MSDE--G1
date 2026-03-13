import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        # 这里的padding参数是一个四元组，分别表示左、上、右、下的padding大小。由于我们只需要在右侧和下方进行padding，所以左和上的padding设置为0。
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

# Compose类是一个组合多个变换的工具类，它接受一个变换列表作为输入，并且在调用时依次应用这些变换。每个变换函数都应该接受图像和标签作为输入，并返回经过变换处理后的图像和标签。Compose类会自动调用每个变换函数，并将变换后的图像和标签传递给下一个变换函数，直到所有的变换都被应用完毕。
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    # t是ransforms列表中的每一个变换函数，t(image, target)表示对图像和标签同时应用这个变换函数，并且将变换后的图像和标签作为输入传递给下一个变换函数。最终，Compose类会返回经过所有变换处理后的图像和标签。
    # 会自动调用__call__方法，所以在使用Compose类时，可以直接将图像和标签作为参数传递给实例对象，例如：transformed_image, transformed_target = compose_instance(image, target)。这样就会依次应用所有的变换函数，并返回最终的变换结果。
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# 随机缩放，使用random.randint()生成一个在min_size和max_size之间的随机整数作为缩放后的大小。然后，使用F.resize函数对图像进行缩放，并且对标签进行相同的缩放操作。需要注意的是，在缩放标签时，使用了插值方法T.InterpolationMode.NEAREST，这是一种最近邻插值方法，可以保持标签的离散性，避免出现模糊的标签值。
class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

# 随机水平翻转，使用random.random()生成一个0到1之间的随机数，如果这个随机数小于给定的翻转概率flip_prob，则对图像和标签进行水平翻转。水平翻转是指将图像沿垂直轴进行镜像翻转，这有助于增加数据的多样性和鲁棒性。
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

# 随机垂直翻转，和随机水平翻转的区别在于，随机垂直翻转是沿着图像的垂直轴进行翻转，而随机水平翻转是沿着图像的水平轴进行翻转。随机垂直翻转可以增加数据的多样性，尤其是在某些任务中，图像的上下方向可能具有不同的特征，例如在医学图像分析中，器官的上下位置可能会影响模型的预测结果。
class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

# 随机裁剪，首先会检查图像的最小边长是否小于给定的裁剪大小，如果是，则会使用pad_if_smaller函数对图像进行padding，以确保图像的最小边长至少为裁剪大小。然后，使用T.RandomCrop.get_params函数随机生成裁剪参数，并使用F.crop函数对图像和标签进行裁剪。
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        # 先计算出随机裁剪的参数，然后再对图像和标签进行裁剪。这样可以确保图像和标签在同一位置被裁剪，从而保持它们之间的对应关系。
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

# 中心裁剪，和随机裁剪的区别在于，中心裁剪是以图像中心为基准进行裁剪，而随机裁剪则是随机选择一个位置进行裁剪。中心裁剪通常用于评估阶段，以确保模型在输入图像的中心区域进行预测，而随机裁剪则用于训练阶段，以增加数据的多样性和鲁棒性。
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

# 转换为张量，并且将标签转换为64位长整型，因为PyTorch的CrossEntropyLoss要求标签必须是64位长整型（torch.int64
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        # PyTorch 的 CrossEntropyLoss 极其死板，它要求作为正确答案的标签，必须是 64 位长整型（torch.int64 或 torch.long
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

# 对图像进行归一化处理，使用给定的均值和标准差对图像进行归一化。归一化是将图像的像素值调整到一个特定的范围内，通常是0到1之间，这有助于加速模型的训练和提高模型的性能。
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target