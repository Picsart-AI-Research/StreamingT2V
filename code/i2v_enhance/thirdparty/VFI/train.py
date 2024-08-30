# Adapted from https://github.com/MCG-NJU/EMA-VFI/blob/main/train.py
import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from Trainer import Model
from dataset import VimeoDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from config import *

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000
        return 2e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5

def train(model, local_rank, batch_size, data_path):
    if local_rank == 0:
        writer = SummaryWriter('log/train_EMAVFI')
    step = 0
    nr_eval = 0
    best = 0
    dataset = VimeoDataset('train', data_path)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('test', data_path)
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    for epoch in range(300):
        sampler.set_epoch(epoch)
        for i, imgs in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step)
            _, loss = model.update(imgs, gt, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss', loss, step)
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss))
            step += 1
        nr_eval += 1
        if nr_eval % 3 == 0:
            evaluate(model, val_data, nr_eval, local_rank)
        model.save_model(local_rank)    
            
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank):
    if local_rank == 0:
        writer_val = SummaryWriter('log/validate_EMAVFI')

    psnr = []
    for _, imgs in enumerate(val_data):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
   
    psnr = np.array(psnr).mean()
    if local_rank == 0:
        print(str(nr_eval), psnr)
        writer_val.add_scalar('psnr', psnr, nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--data_path', type=str, help='data path of vimeo90k')
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    train(model, args.local_rank, args.batch_size, args.data_path)
        
