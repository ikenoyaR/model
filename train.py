import argparse
import os
import csv
import time
import sys
import copy
from collections import OrderedDict


import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
import torchvision
torch.manual_seed(0)


from datasets.dhf1k import DHF1KDataset
from utils.logger import write_train_info, write_csv
from utils.loss import Lossmanager, Postprocess, AverageMeter


def train(args, model):
    """train model"""
    write_train_info(os.path.join(args.save_path, 'train_info.csv'), args)
    train_data = DHF1KDataset(path_data=os.path.join(args.root, 'train'), len_snippet=args.temporal_length, mode='train',
                              mode_sample=args.mode_sampling_input, interval_sampling=args.interval_sampling_input)
    valid_data = DHF1KDataset(path_data=os.path.join(args.root, 'val'), len_snippet=args.temporal_length, mode='valid',
                              mode_sample=args.mode_sampling_input, interval_sampling=args.interval_sampling_input)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.pin_memory, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    dataloaders = {'train':train_dataloader, 'valid':valid_dataloader}
    loss_manager = Lossmanager(args)
    loss_avg_managers = {'train':AverageMeter(), 'valid':AverageMeter()}
    
    params = list(filter(lambda p:p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(params=params, lr=args.lr)
    post_process = Postprocess(output_shape=(360, 640))
    if args.scheduler == None:
        pass
    elif args.scheduler == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=- 1, verbose=False)
    else:
        raise NotImplementedError

    best_loss = sys.float_info.max
    
    torch.save(model, os.path.join(args.save_path, 'init.pth'))
    write_csv(os.path.join(args.save_path, 'loss.csv'), (['epoch', 'train_loss', 'val_loss']), mode='w')


    for epochs in range(args.num_epochs):
        for phase in ['train', 'valid']:
            loss_avg_managers[phase].reset()
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            with tqdm(dataloaders[phase]) as pbar_epoch:
                for video_num, img_num, inputs, labels in pbar_epoch:
                    pbar_epoch.set_description(f'epoch:{epochs}/{args.num_epochs} ({phase})')
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)

                    optimizer.zero_grad()

                    # feed forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        outputs = post_process.transform_predicted_map(outputs, blur=True)
                        assert post_process.confirm_shape(labels), 'predicted map and ground truth map are not same shape'
                        loss = loss_manager.return_loss(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    if args.visualize:
                        torchvision.utils.save_image(inputs.permute(0,2,1,3,4)[0], os.path.join(args.save_path, 'input.png'))
                        torchvision.utils.save_image(labels[0], os.path.join(args.save_path, 'gt.png'))
                        torchvision.utils.save_image(outputs[0], os.path.join(args.save_path, 'output.png'))
                        
                    loss_avg_managers[phase].update(loss.item(), sample=args.batch_size) # when validation, if batchsize set to 1, change this code. 
                    # pbar_epoch.set_postfix(OrderedDict(loss=loss_avg_managers[phase].avg))
                    pbar_epoch.set_postfix(OrderedDict(loss=f'{loss_avg_managers[phase].avg:.2f}'))

            if phase == 'train':
                if args.scheduler != None:
                    scheduler.step()

            if phase == 'valid':
                if loss_avg_managers[phase].avg < best_loss:
                    best_loss = loss_avg_managers[phase].avg 
                    torch.save(model.state_dict(), os.path.join(args.save_path, f'{epochs}_{loss_avg_managers[phase].avg:.5f}.pth'))

                write_csv(os.path.join(args.save_path, 'loss.csv'), ([epochs, loss_avg_managers['train'].avg, loss_avg_managers['valid'].avg]))

