import datetime
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from EfficientViT0 import EfficientViT
from data_util import build_dataset
from threeaugment import new_data_aug_generator
from timm.scheduler import create_scheduler, create_scheduler_v2
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import accuracy
from timm.utils.agc import adaptive_clip_grad
import numpy as np
import os

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EfficientViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')  # 数值稳定防止除零
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 2.5e-2)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')


    # Augmentation parameters
    parser.add_argument('--ThreeAugment', action='store_true')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--finetune', default=False, action='store_true',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/root/autodl-tmp/imagenet100', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET100', choices=['CIFAR', 'IMNET1000', 'IMNET100'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequency of model saving')
    return parser

def train_step(model, train_loader, criterion, optimizer, mixup_fn, clip_grad, opt_eps, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.amp.GradScaler('cuda')

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()

        # 使用 autocast 进行混合精度训练
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()  # 缩放并反向传播，计算梯度
        adaptive_clip_grad(model.parameters(), clip_factor=clip_grad, eps=opt_eps)  # 自适应梯度裁剪

        scaler.step(optimizer)  # 代替optimizer.step()更新参数
        scaler.update()  # 更新缩放因子

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)

        # 获取 targets 的最大索引，用于计算准确率
        targets = targets.argmax(dim=1)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / total
    acc1 = 100. * correct / total
    return avg_loss, acc1


def validate_step(model, valid_loader, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Variants for EfficientViT
def EfficientViT_M0(num_classes,img_size):
    return EfficientViT(num_classes=num_classes,
                        img_size=img_size,
                        embed_dim=[64, 128, 192],
                        depth=[1, 2, 3],
                        num_heads=[4, 4, 4],
                        kernels=[5, 5, 5, 5])

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.dataset.transform = new_data_aug_generator(args)  # 添加3种augmentation

    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size), # 增加样本以获得更稳定的统计结果
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Model, criterion, optimizer, scheduler, mixup_fn
    model = EfficientViT_M0(num_classes=args.nb_classes,img_size=args.input_size).to(device)
    print('number of classes:',args.nb_classes)
    print('number of params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_criterion = SoftTargetCrossEntropy() # mixUp对应软性交叉熵损失
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.opt_eps,
    )
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs,eta_min=args.min_lr)
    '''scheduler = CosineLRScheduler(
        optimizer,
        t_initial=10,  
        t_mul=1,
        lr_min=1e-6,  
        warmup_t=0,  
        warmup_lr_init=1e-6,
        cycle_limit=1,
        t_in_epochs=True
    )'''
    scheduler, _ = create_scheduler(args, optimizer) # default: step on epoch

    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # Open the log file for writing
    with open('training.logs', 'w') as log_file:
        log_file.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n")  # Header

        for epoch in range(args.epochs):
            # Training step
            train_loss, train_acc = train_step(model, train_loader, train_criterion, optimizer,
                                               mixup_fn, args.clip_grad, args.opt_eps, device)
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, lr: {current_lr:.8f}")

            # Validation step
            start_val_time = time.time()
            val_loss, val_acc = validate_step(model, val_loader, device)
            val_time = time.time() - start_val_time
            val_throughput = len(val_loader.dataset) / val_time
            print(f"Epoch {epoch} - Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.2f}%")

            # Save metrics to the log file
            log_file.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc},{val_throughput}\n")

    # torch.save(model.state_dict(), os.path.join(work_path, 'final_model.pth'))
    # torch.save(optimizer.state_dict(), os.path.join(work_path, 'final_optimizer.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'EfficientViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    #if args.output_dir:
    #    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
