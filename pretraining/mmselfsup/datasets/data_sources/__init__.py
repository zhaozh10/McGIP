# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDataSource
from .cifar import CIFAR10, CIFAR100
from .image_list import ImageList
from .imagenet import ImageNet
from .imagenet_21k import ImageNet21k
from .mammo_data_source import MammoDataSource
from .dental_data_source import dentalSource
from .oai import OAIsource
__all__ = [
    'BaseDataSource', 'CIFAR10', 'CIFAR100', 'ImageList', 'ImageNet',
    'ImageNet21k','MammoDataSource','dentalSource', 'OAIsource'
]
