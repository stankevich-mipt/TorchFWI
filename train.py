# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import os
import sys
import time
import datetime
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

import utils
import network
import unet
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as T


from math import exp
from scipy.ndimage import gaussian_filter
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 7, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        
        (_, channel, _, _) = img1.size()

        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1.) / 2.
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return 1. - _ssim(img1, img2, window, self.window_size, channel, self.size_average)

step = 0

def train_one_epoch(model, criterion, optimizer, lr_scheduler, 
                    dataloader, device, epoch, print_freq, writer):
    global step
    model.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data, label in dataloader: #metric_logger.log_every(dataloader, print_freq, header):

        start_time = time.time()

        optimizer.zero_grad()

        guess = torch.clone(label).data.numpy()
        for _ in range(2): guess = gaussian_filter(guess, (0, 0, 5, 5))
        guess = torch.from_numpy(guess).to(device)

        data, label = data.to(device), label.to(device)

       # if np.random.random() < 0.5: guess *= 0.

        output = model(data, guess)
        loss, loss_g1v, loss_g2v = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_g1v_val = loss_g1v.item()
        loss_g2v_val = loss_g2v.item()
        batch_size = data.shape[0]
        metric_logger.update(loss=loss_val, loss_g1v=loss_g1v_val, 
            loss_g2v=loss_g2v_val, lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        if writer:
            writer.add_scalar('loss', loss_val, step)
            writer.add_scalar('loss_g1v', loss_g1v_val, step)
            writer.add_scalar('loss_g2v', loss_g2v_val, step)
        step += 1
        lr_scheduler.step()


def evaluate(model, criterion, dataloader, device, writer):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    with torch.no_grad():
        for data, label in metric_logger.log_every(dataloader, 20, header):

            guess = torch.clone(label).data.numpy()
            for _ in range(5): guess = gaussian_filter(guess, (0, 0, 5, 5))
            guess = torch.from_numpy(guess).to(device)

            data  = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data, guess)
            loss, loss_g1v, loss_g2v = criterion(output, label)
            metric_logger.update(loss=loss.item(), 
                loss_g1v=loss_g1v.item(), 
                loss_g2v=loss_g2v.item())

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
    if writer:
        writer.add_scalar('loss', metric_logger.loss.global_avg, step)
        writer.add_scalar('loss_g1v', metric_logger.loss_g1v.global_avg, step)
        writer.add_scalar('loss_g2v', metric_logger.loss_g2v.global_avg, step)
    return metric_logger.loss.global_avg


def main(args):
    global step

    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    utils.mkdir(args.output_path) # create folder to store checkpoints
    utils.init_distributed_mode(args) # distributed mode initialization

    # Set up tensorboard summary writer
    train_writer, val_writer = None, None
    if args.tensorboard:
        utils.mkdir(args.log_path) # create folder to store tensorboard logs
        if not args.distributed or (args.rank == 0) and (args.local_rank == 0):
            train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
            val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))
                                                                 

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

    # Create dataset and dataloader
    print('Loading data')
    print('Loading training data')
        
    # Normalize data and label to [-1, 1]
    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=args.k), T.log_transform(ctx['data_max'], k=args.k))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(1500., 4500.)
    ])
    if args.train_anno[-3:] == 'txt':
        dataset_train = FWIDataset(
            args.train_anno,
            preload=False,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_data,
            transform_label=transform_label
        )
    else:
        dataset_train = torch.load(args.train_anno)

    print('Loading validation data')
    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            preload=False,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_data,
            transform_label=transform_label
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    print('Creating data loaders')
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_valid)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
    
    model = unet.UNet().to(device)

    #model = network.model_dict[args.model](upsample_mode=args.up_mode, 
    #    sample_spatial=args.sample_spatial, sample_temporal=args.sample_temporal).to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Define loss function
    l1loss   = nn.L1Loss()
    l2loss   = nn.MSELoss()
    ssimloss = SSIM()
    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss     = args.lambda_g1v * loss_g1v +  args.lambda_g2v * loss_g2v + 0.1 * ssimloss(pred, gt)
        return loss, loss_g1v, loss_g2v

    # Scale lr according to effective batch size
    lr = args.lr * args.world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Convert scheduler to be per iteration instead of per epoch
    warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
    lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        lr_scheduler.milestones=lr_milestones
        
    print('Start training')
    start_time = time.time()
    best_loss = 10
    chp=1 
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader_train,
                        device, epoch, args.print_freq, train_writer)
        
        loss = evaluate(model, criterion, dataloader_valid, device, val_writer)
        
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'args': args}
        # Save checkpoint per epoch
        if loss < best_loss:
            utils.save_on_master(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
            print('saving checkpoint at epoch: ', epoch)
            chp = epoch
            best_loss = loss
        # Save checkpoint every epoch block
        print('current best loss: ', best_loss)
        print('current best epoch: ', chp)
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=str, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-t', '--train-anno', default='flatfault_b_train_invnet.txt', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='flatfault_b_val_invnet.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
    parser.add_argument('-l', '--log-path', default='Invnet_models', help='path to parent folder to save logs')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, help='inverse model name')
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    # Training related
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4 , type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')   
    parser.add_argument('-eb', '--epoch_block', type=int, default=40, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=3, help='number of saved block')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')

    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
    
    # Distributed training related
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Tensorboard related
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    
    args.epochs = args.epoch_block * args.num_block

    if args.resume:
        args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
