import time
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
from RASampler import RASampler
from EfficientViT0 import EfficientViT
from data_util import build_dataset
from threeaugment import new_data_aug_generator
from timm.scheduler import create_scheduler, create_scheduler_v2
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils.agc import adaptive_clip_grad
from pathlib import Path
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
                        help='Optimizer (default: adamw)')
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
    parser.add_argument('--threeaugment', action='store_true')
    parser.set_defaults(threeaugment=True)
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
    parser.add_argument('--data-path', default='/root/autodl-tmp/imagenet100', type=str, # 数据路径（需替换）
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET1000', choices=['CIFAR', 'IMNET1000', 'IMNET100'], # 指定数据集（需替换）
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',  # 分布式评估
                        default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # training parameters
    parser.add_argument('--distributed', default=True, action='store_true',
                        help='distributed training')
    parser.add_argument('--world_size', default=8, type=int,  # 需修改
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--resume', default='', help='resume from checkpoint') # 需要指定ckpt路径
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_freq', default=50, type=int,
                        help='frequency of model saving')
    return parser

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    # 设置 GPU 设备
    torch.cuda.set_device(args.gpu)

    # 初始化分布式通信
    dist.init_process_group(
        backend='nccl',  # 使用 NCCL 后端进行 GPU 通信
        init_method='env://', # 自动从环境变量解析
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()  # 同步所有进程
    print(f'| distributed init (rank {args.rank}): {args.gpu}', flush=True)
    setup_for_distributed(args.rank == 0)  # 输出控制

def train_step(model, train_loader, criterion, optimizer, mixup_fn, scaler, clip_grad, opt_eps, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = mixup_fn(inputs, targets)

        # 使用 autocast 进行混合精度训练
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()  # 缩放并反向传播，计算梯度
        adaptive_clip_grad(model.parameters(), clip_factor=clip_grad, eps=opt_eps)  # 自适应梯度裁剪

        scaler.step(optimizer)  # 代替optimizer.step()更新参数
        scaler.update()  # 更新缩放因子
        optimizer.zero_grad()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)

        # 获取 targets 的最大索引，用于计算准确率
        targets = targets.argmax(dim=1)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / total
    acc1 = 100. * correct / total
    return avg_loss, acc1


def validate(model, valid_loader, device):
    model.eval()
    val_loss = 0.0
    correct_top1 = 0  # Top-1 正确数
    correct_top5 = 0  # Top-5 正确数
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)

            # Top-1 和 Top-5 的预测值
            _, predicted_top5 = outputs.topk(5, dim=1)  # 获取每行的前5个预测值
            total += targets.size(0)

            # Top-1：检查第一个预测是否正确
            correct_top1 += predicted_top5[:, 0].eq(targets).sum().item()

            # Top-5：检查前5个预测中是否包含正确答案
            correct_top5 += torch.sum(predicted_top5.eq(targets.view(-1, 1))).item()

    avg_loss = val_loss / total
    acc1 = 100. * correct_top1 / total  # Top-1 Accuracy
    acc5 = 100. * correct_top5 / total  # Top-5 Accuracy
    return avg_loss, acc1, acc5


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

    if args.distributed:  # 是否分布式训练
        init_distributed_mode(args)
        num_tasks = dist.get_world_size()
        print(num_tasks)
        global_rank = dist.get_rank()
        print(global_rank)
        if args.repeated_aug: # 是否重复数据增强
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:  # 是否分布式验证
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
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

    # Model, criterion, optimizer, scheduler, scaler, mixup_fn
    model = EfficientViT_M0(num_classes=args.nb_classes,img_size=args.input_size).to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

        linear_scaled_lr = args.lr * args.batch_size * dist.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    print('number of classes:',args.nb_classes)
    print('number of params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_criterion = SoftTargetCrossEntropy() # mixUp对应软性交叉熵损失
    optimizer = optim.AdamW(
        model_without_ddp.parameters(), #  model.parameters()
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.opt_eps,
    )
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs,eta_min=args.min_lr)
    #scheduler = CosineLRScheduler(optimizer,t_initial=10,t_mul=1,lr_min=1e-6,warmup_t=0,warmup_lr_init=1e-6,cycle_limit=1,t_in_epochs=True)
    scheduler, _ = create_scheduler(args, optimizer) # default: step on epoch
    scaler = torch.amp.GradScaler('cuda')
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.nb_classes)

    output_dir = Path(args.output_dir)
    if args.output_dir and args.rank == 0:
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")
    if args.resume:
        print("Loading local checkpoint at {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'])
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])

    ######################################## training ############################################
    print(f"Start training for {args.epochs} epochs")

    if args.rank == 0:  # 主进程
        log_file = open('training.logs', 'w')
        log_file.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc@1,Val Acc@5\n")  # Header

    for epoch in range(args.start_epoch,args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_step(model, train_loader, train_criterion, optimizer,
                                           scaler, mixup_fn, args.clip_grad, args.opt_eps, device)
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        start_val_time = time.time()
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, device)
        validation_time = time.time() - start_val_time
        val_throughput = len(val_loader.dataset) / validation_time

        if args.rank == 0:  # 主进程打印和写日志
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, lr: {current_lr:.8f}")
            print(f"Epoch {epoch} - Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc1:.2f}%, Val Throughput: {val_throughput:.2f} img/s")
            log_file.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc1},{val_acc5},{val_throughput}\n")

            if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(output_dir, 'checkpoint_' + str(epoch) + 'epoch.pth')
                    checkpoint_paths = [ckpt_path]
                    print("Saving checkpoint to {}".format(ckpt_path))
                    for checkpoint_path in checkpoint_paths:
                        torch.save({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            'scaler': scaler.state_dict(),
                            'args': args,
                            }, checkpoint_path)

    if args.rank == 0:  # 关闭日志
        log_file.close()
        print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'EfficientViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)