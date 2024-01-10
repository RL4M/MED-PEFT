# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from timm.data import create_transform
from timm.data.random_erasing import RandomErasing
from torchvision.datasets.folder import default_loader,VisionDataset,IMG_EXTENSIONS
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import codecs
import torch
from torch import nn
import random
import numpy as np
import pydicom
import cv2
from PIL import Image


np.random.seed(0)


class MultiLabelDatasetFolder(VisionDataset):
    def __init__(
            self,
            args,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
    ) -> None:
        super(MultiLabelDatasetFolder, self).__init__(root, transform=transform)
        samples = self.read_samples(self.root)
        self.args = args
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    def read_samples(self,root):
        data = codecs.open(root,"r","utf-8","ignore")
        outputList = []
        for line in data:
            outputList.append(line.strip())
        path_list = []
        for output in outputList:
            if 'NIH' in root:
                path_list.append((os.path.join(''.join(root.split('_')[:-1]),'all_classes',output.split(' ')[0]),[ int(i) for i in output.split(' ')[1:]]))
            else:
                path_list.append((os.path.join('/'.join(root.split('/')[:-1]),output.split(' ')[0]),[ int(i) for i in output.split(' ')[1:]]))
        return path_list

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = torch.FloatTensor(target)

        if 'RSNA' not in self.args.data_path:
            sample = self.loader(path)
        else:
            dcm = pydicom.read_file(path)
            x = dcm.pixel_array

            x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                x = cv2.bitwise_not(x)
            sample = Image.fromarray(x).convert("RGB")
                
        if self.transform is not None:
            sample = self.transform(sample)

        return target, sample

    def __len__(self) -> int:
        return len(self.samples)

def build_dataset(is_train, args):
    if args.optimizer == 'sgd':
        transform = build_transform_threesamechannel_for_sgd(is_train, args)
    else:
        transform = build_transform_threesamechannel(is_train, args)
    print(transform)
    if is_train:
        if args.data_size == '1%':
            filename = 'train_1.txt'
        elif args.data_size == '10%':
            filename = 'train_10.txt'
        elif args.data_size == '100%':
            filename = 'train_list.txt'
    else:
        if args.eval:
            filename = 'test_list.txt'
        else:
            filename = 'val_list.txt'
    
    root = os.path.join(args.data_path, filename)
    
    dataset = MultiLabelDatasetFolder(args, root, default_loader, IMG_EXTENSIONS, transform=transform)
    print(dataset)

    return dataset


def build_transform_threesamechannel(is_train, args):
    if 'MRM' in args.finetune.split('/')[-1]:
        mean=[0.4978]
        std=[0.2449]
    else:
        mean=[0.4785]
        std=[0.2834]
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        transform.transforms.insert(3,transforms.Grayscale(num_output_channels=3))
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.Grayscale(num_output_channels=3))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_transform_threesamechannel_for_sgd(is_train, args):
    if 'MRM' in args.finetune.split('/')[-1]:
        mean=[0.4978]
        std=[0.2449]
    else:
        mean=[0.4785]
        std=[0.2834]
    # train transform
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop((args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
    # valid & test transform
    else:
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            transforms.CenterCrop((args.input_size, args.input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
    return transform
