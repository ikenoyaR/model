import os


import torch


from model.ViNet import ViNet
from model.S3D import load_S3D


def load_model(args):
    """
    Define a model and load weights.
    The model is sent to the device specified in args.device.
    """
    if args.model == 'ViNet':
        model = ViNet(
            num_hier=3,
            num_clips=args.temporal_length
        )
        model = model.to('cpu')
        if args.path_load_weight != '':
            assert os.path.exists(args.path_load_weight), f"Path:{args.path_load_weight} doesn't exists."
            print(f'load {args.path_load_weight}')
            model.load_state_dict(torch.load(args.path_load_weight))
        elif args.path_backbone_pretrained:
            model = load_S3D(model, args.path_backbone_pretrained)
        else:
            print('Model does not load any trained weights.\n')

    else:
        raise NotImplementedError
    
    return model.to(args.device)