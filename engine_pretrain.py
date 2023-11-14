import math
import numpy as np
import torch
from utils.misc import MetricLogger, SmoothedValue
from utils.misc import mae_loss, print_rank_0

def train_one_epoch(args,
                    device,
                    model,
                    data_loader,
                    optimizer,
                    epoch,
                    lf,
                    loss_scaler,
                    local_rank):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    epoch_size = len(data_loader)

    # train one epoch
    for iter_i, (images, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = iter_i + epoch * epoch_size
        nw = args.wp_epoch * epoch_size
        # Warmup
        if ni <= nw:
            xi = [0, nw]  # x interp
            for x in optimizer.param_groups:
                x['lr'] = np.interp(ni, xi, [0.0, x['initial_lr'] * lf(epoch)])

        # To device
        images = images.to(device, non_blocking=True)

        # Inference
        with torch.cuda.amp.autocast():
            ## forward
            output = model(images)
            ## compute loss
            loss = mae_loss(images, output['x_pred'], output['mask'], args.patch_size, args.norm_pix_loss)

        # Check loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backward & Optimize
        loss /= args.grad_accumulate
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(iter_i + 1) % args.grad_accumulate == 0)
        if ni % args.grad_accumulate == 0:
            optimizer.zero_grad()

        if args.cuda:
            torch.cuda.synchronize()

        # Logs
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_rank_0("Averaged stats: {}".format(metric_logger), local_rank)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}