import argparse
import os
import csv
import time


import cv2
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
torch.manual_seed(0)


from train import train
from test import test
from predict import predict
from model.utils import load_model



def main():
    parser = argparse.ArgumentParser()
    """mode choice [train, test]"""
    parser.add_argument("--model", type=str, default='ViNet', choices=['ViNet'], help="model name")

    parser.add_argument("--root", type=str, default='datasets/DHF1K', help="path to dataset")
    parser.add_argument("--dataset", type=str, default='DHF1K', help= 'name of dataset')

    parser.add_argument("--path_backbone_pretrained", type=str, default='./checkpoints/S3D_kinetics400.pt', help="path to pretrained backbone weight")
    parser.add_argument("--path_load_weight", type=str, default='', help="path to model checkpoints")

    # training parameter
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs you want to train")
    parser.add_argument("--batch_size", type=int, default=2, help="number of batch_size")
    parser.add_argument("--workers", type=int, default=4, help="number of threads")
    parser.add_argument("--pin_memory", action='store_false', help="argument of PyTorch Dataloader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--scheduler", type=str, default=None, help="Learning rate sheduler")
    parser.add_argument('--device', default='cuda', help='gpu devices')
    
    # evaluation metric
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--cc', action='store_true')
    parser.add_argument('--nss', action='store_true')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--ig', action='store_true')
    parser.add_argument('--kl_coeff',default=1.0, type=float)
    parser.add_argument('--cc_coeff',default=-1.0, type=float)
    parser.add_argument('--nss_coeff',default=-1.0, type=float)
    parser.add_argument('--sim_coeff',default=-1.0, type=float)
    parser.add_argument('--ig_coeff',default=-1.0, type=float)
    parser.add_argument("--root_sauc_other_map", type=str, default='', help="for evaluation sauc in DHF1K")
    parser.add_argument("--root_ig_baseline", type=str, default='', help="center prior map 'baseline/centerprior_map.png'")

    # input
    parser.add_argument("--image_width", type=int, default=384, help="image width")
    parser.add_argument("--image_height", type=int, default=224, help="image height")
    parser.add_argument("--temporal_length", type=int, default=32, help="Temporal dimension")
    parser.add_argument("--mode_sampling_input", type=str, default='sliding_window', help="sampling_mode")
    parser.add_argument("--interval_sampling_input", type=int, default=1, help="sampling interval of input frame")

    # options
    parser.add_argument("--visualize", action='store_true', help='visualize input, gt, output')
    parser.add_argument("--nosave", action='store_true', help='do not save saliency map')
    parser.add_argument("--save_heatmap", action='store_true', help='save heatmap')
    parser.add_argument('--mode', default='train', choices=['train', 'test', 'predict'], help='choose modes train or test(evaluate) or predict(saliency maps)')
    parser.add_argument('--project', default='results/train', help='save results to project/name')
    parser.add_argument('--name', default='predict', help='save results to project/name')

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.save_path = os.path.join(args.project, args.name)
    if not os.path.exists(args.project):
        os.makedirs(args.project)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    else:
        print(f'save path : {args.save_path}')

    model = load_model(args)

    if args.mode == 'train':
        train(args, model)
    elif args.mode =='test':
        test(args, model)
    elif args.mode == 'predict':
        predict(args, model)


if __name__ == '__main__':
    main()