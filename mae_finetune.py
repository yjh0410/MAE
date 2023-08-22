from copy import deepcopy
import os
import time
import math
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import build_dataset, build_dataloader, build_transform
from models import build_model
from utils import distributed_utils
from utils.misc import setup_seed, accuracy
from utils.com_flops_params import FLOPs_and_Params


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size.')    
    parser.add_argument('--img_dim', type=int, default=3,
                        help='3 for RGB; 1 for Gray.')    
    parser.add_argument('--patch_size', type=int, default=16,
                        help='patch_size.')    
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size on all GPUs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--path_to_save', type=str, default='weights/',
                        help='path to save trained model.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    # Epoch
    parser.add_argument('--wp_epoch', type=int, default=5, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='start epoch for finetune with MAE pretrained')
    parser.add_argument('--max_epoch', type=int, default=50, 
                        help='max epoch')
    parser.add_argument('--eval_epoch', type=int, default=1, 
                        help='max epoch')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    # Model
    parser.add_argument('-m', '--model', type=str, default='vit_tiny',
                        help='model name')
    parser.add_argument('--mae_pretrained', default=None, type=str,
                        help='load MAE pretrained weight.')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema.')
    # Optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='adamw',
                        help='sgd, adam')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='learning rate for training model')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='the final lr')
    parser.add_argument('-accu', '--grad_accumulate', type=int, default=1,
                        help='gradient accumulation')
    # DDP
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()

    
def main():
    args = parse_args()
    print(args)
    # set random seed
    setup_seed(args.seed)

    path_to_save = os.path.join(args.path_to_save, args.model)
    os.makedirs(path_to_save, exist_ok=True)
    
    # ------------------------- Build DDP environment -------------------------
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # ------------------------- Build CUDA -------------------------
    if args.cuda:
        print("use cuda")
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ------------------------- Build Tensorboard -------------------------
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)
        tblogger = SummaryWriter(log_path)

    # ------------------------- Build Transforms -------------------------
    train_transform = build_transform(args, is_train=True)
    val_transform = build_transform(args, is_train=False)

    # ------------------------- Build Dataset -------------------------
    train_dataset = build_dataset(args, train_transform, is_train=True)
    val_dataset = build_dataset(args, val_transform, is_train=False)

    # ------------------------- Build Dataloader -------------------------
    train_dataloader = build_dataloader(args, train_dataset, is_train=True)
    val_dataloader = build_dataloader(args, val_dataset, is_train=False)

    print('=================== Dataset Information ===================')
    print('Train dataset size : ', len(train_dataset))
    print('Val dataset size   : ', len(val_dataset))

    # ------------------------- Build Model -------------------------
    model = build_model(args)
    model.train().to(device)
    print(model)
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model)
        model_copy.eval()
        FLOPs_and_Params(model=model_copy, size=args.img_size)
        model_copy.train()
        del model_copy
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # ------------------------- Build DDP Model -------------------------
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ------------------------- Train Config -------------------------
    best_acc1 = -1.
    epoch_size = len(train_dataloader)

    # ------------------------- Build Optimzier -------------------------
    args.base_lr = args.base_lr / 256 * args.batch_size * args.grad_accumulate  # auto scale lr
    args.min_lr = args.min_lr / 256 * args.batch_size * args.grad_accumulate    # auto scale lr
    optimizer = optim.AdamW(model_without_ddp.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    # ------------------------- Build Lr Scheduler -------------------------
    lf = lambda x: ((1 - math.cos(x * math.pi / args.max_epoch)) / 2) * (args.min_lr / args.base_lr - 1) + 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # ------------------------- Build Criterion -------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # ------------------------- Training Pipeline -------------------------
    t0 = time.time()
    print("=================== Start training ===================")
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        for iter_i, (images, target) in enumerate(train_dataloader):
            ni = iter_i + epoch * epoch_size
            nw = args.wp_epoch * epoch_size
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                for x in optimizer.param_groups:
                    x['lr'] = np.interp(ni, xi, [0.0, x['initial_lr'] * lf(epoch)])

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Inference
            output = model(images)

            # Loss
            loss = criterion(output, target)

            # Accuracy
            acc = accuracy(output, target, topk=(1, 5,))            

            # Backward
            loss /= args.grad_accumulate
            loss.backward() 

            # Update
            if ni % args.grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Logs
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('loss',  loss.item() * args.grad_accumulate,  ni)
                    tblogger.add_scalar('acc1',  acc[0].item(),  ni)
                    tblogger.add_scalar('acc5',  acc[1].item(),  ni)
                
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch + 1, args.max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}]'.format(cur_lr[0])
                # loss infor
                log += '[loss: {:.6f}]'.format(loss.item() * args.grad_accumulate)
                # other infor
                log += '[acc1: {:.2f}]'.format(acc[0].item())
                log += '[acc5: {:.2f}]'.format(acc[1].item())
                log += '[time: {:.2f}]'.format(t1 - t0)

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()

        # Evaluate
        if distributed_utils.is_main_process():
            if (epoch % args.eval_epoch) == 0 or (epoch == args.max_epoch - 1):
                print('evaluating ...')
                loss, acc1 = validate(device, val_dataloader, model_without_ddp, criterion)
                print('Eval Results: [loss: %.2f][acc1: %.2f]' % (loss.item(), acc1[0].item()), flush=True)

                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                if is_best:
                    print('saving the model ...')
                    weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.model, epoch, acc1[0].item())
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save({'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'acc1': acc1[0].item(),
                                'epoch': epoch},
                                checkpoint_path)                      

        lr_scheduler.step()


def validate(device, val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    acc1_num_pos = 0.
    count = 0.
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i % 100 == 0:
                print("[%d]/[%d] ..." % (i, len(val_loader)))
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # inference
            output = model(images)

            # loss
            loss = criterion(output, target)

            # accuracy
            cur_acc1 = accuracy(output, target, topk=(1,))

            # Count the number of positive samples
            bs = images.shape[0]
            count += bs
            acc1_num_pos += cur_acc1[0] * bs
        
        # top1 acc
        acc1 = acc1_num_pos / count

    # switch to train mode
    model.train()

    return loss, acc1


if __name__ == "__main__":
    main()