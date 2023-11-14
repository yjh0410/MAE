import os
import cv2
import time
import math
import datetime
import argparse
import numpy as np
from copy import deepcopy

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- Torchvision compoments ----------------
import torchvision.transforms as transforms

# ---------------- Dataset compoments ----------------
from data import build_dataset, build_dataloader
from models import build_model

# ---------------- Utils compoments ----------------
from utils import distributed_utils
from utils.misc import setup_seed, modify_optimizer
from utils.misc import load_model, save_model
from utils.misc import unpatchify, mae_loss, print_rank_0
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.com_flops_params import FLOPs_and_Params

# ---------------- Training engine ----------------
from engine_pretrain import train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size.')    
    parser.add_argument('--img_dim', type=int, default=3,
                        help='3 for RGB; 1 for Gray.')    
    parser.add_argument('--patch_size', type=int, default=16,
                        help='patch_size.')    
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='mask ratio.')    
    # Basic
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
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
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate model.')
    # Epoch
    parser.add_argument('--wp_epoch', type=int, default=40, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='start epoch for finetune with MAE pretrained')
    parser.add_argument('--eval_epoch', type=int, default=20, 
                        help='warmup epoch for finetune with MAE pretrained')
    parser.add_argument('--max_epoch', type=int, default=400, 
                        help='max epoch')
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    # Model
    parser.add_argument('-m', '--model', type=str, default='vit_nano',
                        help='model name')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    # Optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='adamw',
                        help='sgd, adam')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='learning rate for training model')
    parser.add_argument('--min_lr', type=float, default=0,
                        help='the final lr')
    parser.add_argument('-accu', '--grad_accumulate', type=int, default=1,
                        help='gradient accumulation')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Clip gradient norm (default: None, no clipping)')
    # Loss
    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='normalize pixels before computing loss.')
    # DDP
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')

    return parser.parse_args()

    
def main():
    args = parse_args()
    # set random seed
    setup_seed(args.seed)

    path_to_save = os.path.join(args.path_to_save, args.dataset, args.model)
    os.makedirs(path_to_save, exist_ok=True)
    
    # ------------------------- Build DDP environment -------------------------
    local_rank = local_process_rank = -1
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
        try:
            # Multiple Mechine & Multiple GPUs (world size > 8)
            local_rank = torch.distributed.get_rank()
            local_process_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))
        except:
            # Single Mechine & Multiple GPUs (world size <= 8)
            local_rank = local_process_rank = torch.distributed.get_rank()

    print_rank_0(args, local_rank)
    # ------------------------- Build CUDA -------------------------
    if args.cuda:
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ------------------------- Build Tensorboard -------------------------
    tblogger = None
    if local_rank <= 0 and args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        time_stamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, time_stamp)
        os.makedirs(log_path, exist_ok=True)
        tblogger = SummaryWriter(log_path)

    # ------------------------- Build Transforms -------------------------
    train_transform = None
    if 'cifar' not in args.dataset:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # ------------------------- Build Dataset -------------------------
    train_dataset = build_dataset(args, transform=train_transform, is_train=True)

    # ------------------------- Build Dataloader -------------------------
    train_dataloader = build_dataloader(args, train_dataset, is_train=True)
    epoch_size = len(train_dataloader)

    print_rank_0('=================== Dataset Information ===================', local_rank)
    print_rank_0('Train dataset size : {}'.format(len(train_dataset)), local_rank)

    # ------------------------- Build Model -------------------------
    model = build_model(args)
    model.train().to(device)
    if local_rank <= 0:
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
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ------------------------- Build Optimzier -------------------------
    ## learning rate
    args.base_lr = args.base_lr / 256 * args.batch_size * args.grad_accumulate  # auto scale lr
    args.min_lr  = args.min_lr  / 256 * args.batch_size * args.grad_accumulate    # auto scale lr
    ## modified optimizer
    optimizer = modify_optimizer(model_without_ddp, args.base_lr, args.weight_decay)

    # ------------------------- Build Loss scaler -------------------------
    loss_scaler = NativeScaler()
    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # ------------------------- Build Lr Scheduler -------------------------
    lf = lambda x: ((1 - math.cos(x * math.pi / args.max_epoch)) / 2) * (args.min_lr / args.base_lr - 1) + 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # ------------------------- Eval before Train Pipeline -------------------------
    if args.eval:
        print('visualizing ...')
        visualize(args, device, model_without_ddp)
        exit(0)

    # ------------------------- Training Pipeline -------------------------
    start_time = time.time()
    print_rank_0("=============== Start training for {} epochs ===============".format(args.max_epoch), local_rank)
    for epoch in range(args.start_epoch, args.max_epoch):
        # Train one epoch
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(args, device, model, train_dataloader, optimizer, epoch, lf, loss_scaler, local_rank, tblogger)

        # LR scheduler
        lr_scheduler.step()

        # Evaluate
        if local_rank <= 0 and (epoch % args.eval_epoch == 0 or epoch + 1 == args.epochs):
            print('- saving the model after {} epochs ...'.format(epoch))
            save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                       optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_rank_0('Training time {}'.format(total_time_str), local_rank)


def visualize(args, device, model):
    # test dataset
    val_dataset = build_dataset(args, is_train=False)
    val_dataloader = build_dataloader(args, val_dataset, is_train=False)

    # save path
    save_path = "vis_results/{}/{}".format(args.dataset, args.model)
    os.makedirs(save_path, exist_ok=True)

    # switch to evaluate mode
    model.eval()
    patch_size = args.patch_size
    pixel_mean = val_dataloader.dataset.pixel_mean
    pixel_std  = val_dataloader.dataset.pixel_std
    with torch.no_grad():
        for i, (images, target) in enumerate(val_dataloader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # inference
            output = model(images)
            # loss
            loss = mae_loss(images, output['x_pred'], output['mask'], patch_size, model.norm_pix_loss)

            # denormalize input image
            org_img = images[0].permute(1, 2, 0).cpu().numpy()
            org_img = (org_img * pixel_std + pixel_mean) * 255.
            org_img = org_img.astype(np.uint8)

            # masked image
            mask = output['mask'].unsqueeze(-1).repeat(1, 1, patch_size**2 *3)  # [B, H*W] -> [B, H*W, p*p*3]
            mask = unpatchify(mask, patch_size)
            mask = mask[0].permute(1, 2, 0).cpu().numpy()
            masked_img = org_img * (1 - mask)  # 1 is removing, 0 is keeping
            masked_img = masked_img.astype(np.uint8)

            # denormalize reconstructed image
            pred_img = unpatchify(output['x_pred'], patch_size)
            pred_img = pred_img[0].permute(1, 2, 0).cpu().numpy()
            pred_img = (pred_img * pixel_std + pixel_mean) * 255.
            pred_img = org_img * (1 - mask) + pred_img * mask
            pred_img = pred_img.astype(np.uint8)

            # visualize
            vis_image = np.concatenate([masked_img, org_img, pred_img], axis=1)
            vis_image = vis_image[..., (2, 1, 0)]
            print("[{}]/[{}] | Label: {} | Loss: {:.4f} ".format(i, len(val_dataloader), int(target[0]), loss.item()))
            cv2.imshow('masked | origin | reconstruct ', vis_image)
            cv2.waitKey(0)

            # save
            cv2.imwrite('{}/{:06}.png'.format(save_path, i), vis_image)


if __name__ == "__main__":
    main()