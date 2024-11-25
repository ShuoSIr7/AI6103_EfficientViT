import time
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
from RASampler import RASampler
from model.EfficientViT import EfficientViT # original
from model.model import MyEfficientViT # ours
from data_util import build_dataset
from threeaugment import new_data_aug_generator
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import NativeScaler,accuracy
from timm.utils.agc import adaptive_clip_grad
from pathlib import Path
import os
import json

def get_args_parser():
    parser = argparse.ArgumentParser(
        'EfficientViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=512, type=int) # this should be modified to 2048//world_size
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
    parser.add_argument('--threeaugment', action='store_true', default=True)
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true',default=True)

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

    # Dataset parameters
    parser.add_argument('--data-path', default='/root/autodl-tmp/imagenet100', type=str, # data path
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET100', choices=['IMNET1000', 'IMNET100'], # dataset
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eval', action='store_true',default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',default=False, # 修改
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # training parameters
    parser.add_argument('--distributed', action='store_true', #是否分布式
                        help='distributed training')
    parser.add_argument('--world_size', default=1, type=int,  # can be parsed from torchrun --nproc_per_node
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--resume', default='', help='resume from checkpoint') # ckpt path is needed for resume
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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

def train_step(model, train_loader, criterion, optimizer, mixup_fn,
               loss_scaler, clip_grad, opt_eps, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device,non_blocking=True), targets.to(device,non_blocking=True)
        inputs, targets = mixup_fn(inputs, targets) # apply mixUp

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # 缩放 + 反向传播（计算梯度）
        loss_scaler.scale(loss).backward()

        # 反缩放 + 梯度裁剪
        loss_scaler.unscale_(optimizer)
        adaptive_clip_grad(model.parameters(), clip_factor=clip_grad, eps=opt_eps)  # 显式调用，自适应梯度裁剪

        # 代替optimizer.step()更新参数
        loss_scaler.step(optimizer)
        loss_scaler.update()  # 调整缩放因子

        # 进程同步
        torch.cuda.synchronize()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)

        # 获取 targets 的最大索引，用于计算准确率
        targets = targets.argmax(dim=1)
        correct += predicted.eq(targets).sum().item()

    # 汇总所有进程的 loss 和正确样本数
    train_loss_tensor = torch.tensor([train_loss], dtype=torch.float32, device=device)
    correct_tensor = torch.tensor([correct], dtype=torch.float32, device=device)
    total_tensor = torch.tensor([total], dtype=torch.float32, device=device)

    #dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
    #dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    #dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    # 计算全局平均损失和准确率
    batch_loss = train_loss_tensor.item() / total_tensor.item()
    batch_acc1 = 100.0 * correct_tensor.item() / total_tensor.item()

    return batch_loss, batch_acc1


def validate(model, valid_loader, device):
    model.eval()
    val_loss = 0.0
    correct_top1 = 0  # Top-1 count
    correct_top5 = 0  # Top-5 count
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            # accumulate correct counts
            correct_top1 += acc1.item() * inputs.size(0)
            correct_top5 += acc5.item() * inputs.size(0)
            total += targets.size(0)

    torch.cuda.synchronize()  # synchronize between process

    val_loss_tensor = torch.tensor([val_loss], dtype=torch.float32, device=device)
    correct_top1_tensor = torch.tensor([correct_top1], dtype=torch.float32, device=device)
    correct_top5_tensor = torch.tensor([correct_top5], dtype=torch.float32, device=device)
    total_tensor = torch.tensor([total], dtype=torch.float32, device=device)

    # aggregate between process
    #dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    #dist.all_reduce(correct_top1_tensor, op=dist.ReduceOp.SUM)
    #dist.all_reduce(correct_top5_tensor, op=dist.ReduceOp.SUM)
    #dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    # calculate on batch
    batch_loss = val_loss_tensor.item() / total_tensor.item()
    batch_acc1 = correct_top1_tensor.item() / total_tensor.item()
    batch_acc5 = correct_top5_tensor.item() / total_tensor.item()

    return batch_loss, batch_acc1, batch_acc5

# Variants for EfficientViT
def EfficientViT_M0(num_classes,img_size):
    '''return EfficientViT(num_classes=num_classes,
                        img_size=img_size,
                        embed_dim=[64, 128, 192],
                        depth=[1, 2, 3],
                        num_heads=[4, 4, 4],
                        kernels=[5, 5, 5, 5])'''
    #return EfficientViT(num_classes=num_classes)  # if ours, call MyEfficientViT
    return MyEfficientViT(num_classes=num_classes)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    print("distributed mode:", args.distributed)
    if args.distributed:
        init_distributed_mode(args)
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            #print('dist_val success')
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        args.rank = 0

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    train_loader.dataset.transform = new_data_aug_generator(args)  # add three augmentation

    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size), # 1.5 x samples to stablize validation results
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Model, criterion, optimizer, scheduler, scaler, mixup_fn
    model = EfficientViT_M0(num_classes=args.nb_classes,img_size=args.input_size).to(device)
    print('model device:',device)
    print('number of classes:',args.nb_classes)
    print('number of params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.distributed:
        model_without_ddp = model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

        linear_scaled_lr = args.lr * args.batch_size * dist.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    #loss_scaler = torch.cuda.amp.GradScaler()
    loss_scaler = torch.amp.GradScaler('cuda') # different version

    train_criterion = SoftTargetCrossEntropy() # Soft for mixUp
    optimizer = optim.AdamW(
        model.parameters(), #  model_without_ddp.parameters()
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.opt_eps,
    )

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,  # all epochs as a cycle
        lr_min=args.min_lr,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.warmup_lr,
        cycle_limit=1,
        t_in_epochs=True # default: step on epoch
    )
    #scheduler, _ = create_scheduler(args, optimizer) # default: step on epoch

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
            if 'loss_scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['loss_scaler'])
    ######################################## training ############################################
    print(f"Start training for {args.epochs} epochs")

    if args.rank == 0:
        log_file = open('training.logs', 'w')
        log_file.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc@1,Val Acc@5\n")  # Log Header

    for epoch in range(args.start_epoch,args.epochs):
        if args.distributed: # disorder sampling sequence of different process
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_step(model, train_loader, train_criterion, optimizer, mixup_fn,
                                           loss_scaler, args.clip_grad, args.opt_eps, device)

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        start_val_time = time.time()
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, device)
        validation_time = time.time() - start_val_time
        val_throughput = len(val_loader.dataset) / validation_time

        if args.rank == 0:  # if master process
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, lr: {current_lr:.8f}")
            print(f"Epoch {epoch} - Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc1:.2f}%, Val Throughput: {val_throughput:.2f} img/s")
            log_file.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc1},{val_acc5},{val_throughput}\n")
            if epoch == 100 or epoch == 200 or epoch == args.epochs - 1:
                ckpt_path = os.path.join(output_dir, 'checkpoint_' + str(epoch) + 'epoch.pth')
                checkpoint_paths = [ckpt_path]
                print("Saving checkpoint to {}".format(ckpt_path))
                for checkpoint_path in checkpoint_paths:
                    torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'loss_scaler': loss_scaler.state_dict(),
                    'args': args,
                    }, checkpoint_path)

    if args.rank == 0:
        log_file.close()
        print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'EfficientViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("===Running Success===")
    main(args)
