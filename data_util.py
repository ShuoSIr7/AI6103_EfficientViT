import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
#from timm.data import TimmDatasetTar
from timm.data import ImageDataset as TimmDatasetTar

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET1000':
        prefix = 'train' if is_train else 'val'
        data_dir = os.path.join(args.data_path, f'ILSVRC2012_img_{prefix}.tar')
        if os.path.exists(data_dir):
            dataset = TimmDatasetTar(data_dir, transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET100':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if args.finetune:
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                              interpolation=3)
        )
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
