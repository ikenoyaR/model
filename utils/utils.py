import os


import cv2
import torch
import numpy as np


def load_npy(path='baseline/unnormed_centerprior_map.npy', type='float') -> torch.tensor:
    """
    load npy file and return torch.tensor
    type choices ['float', 'int']
    """
    tensor = np.load(path)
    if type == 'float':
        tensor = torch.from_numpy(tensor.astype(np.float32)).clone()
    elif type == 'int':
        tensor =  torch.from_numpy(tensor.astype(np.float32)).clone()
    else:
        raise NotImplementedError
    return tensor

def normalize(img, regularize=False):
    if not regularize:
        normed_img = (img - img.min())/(img.max()-img.min()*1.0)
    else:
        eps = 2.2204e-16
        normed_img = (img - img.min())/(img.max()-img.min()*1.0 + eps)
    return normed_img

def normalize_map(s_map, regularize=False):
    eps = 1e-7 if regularize else 0
    if len(s_map.size()) == 3:
        """assume (B, H, W)"""
        # normalize the salience map (as done in MIT code)
        batch_size = s_map.size(0)
        h = s_map.size(1)
        w = s_map.size(2)

        min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)
        max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, h, w)
    elif len(s_map.size()) == 2:
        """assume (H, W)"""
        min_s_map = s_map.min()
        max_s_map = s_map.max()
    else:
        raise NotImplementedError
    norm_s_map = (s_map - min_s_map)/(max_s_map-min_s_map*1.0 + eps)
    return norm_s_map


def get_img(path_img, mode='img'):
	"""
	mode choices:['img', 'map', 'fix']
	return img (ndarray)
	if mode == 'img':
		shape of img is (height, width, channel)
	elif mode == 'map':
		shape of img is (height, width)
	elif mode == 'fix':
		shape of img is (height, width)
	"""
	if mode == 'img':
		img = cv2.imread(path_img, 1)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	elif mode == 'map':
		img = cv2.imread(path_img, 1)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	elif mode == 'fix':
		img = cv2.imread(path_img, 1)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		raise NotImplementedError
	return img

    
def get_ig_baseline(path):
    assert os.path.exists(path), 'IG baseline (center prior map) does not exist'
    if path.split('.')[-1] == 'npy':
        center_prior = np.load(path)
    else:
        center_prior = get_img(path, mode='map')
    if center_prior.max() != 1.0 or center_prior.min() != 0.0:
        print('IG baseline Min-Max normalization...')
    assert (center_prior.max()-center_prior.min()), 'center_prior may be zero'
    center_prior = (center_prior-center_prior.min())/(center_prior.max()-center_prior.min())
    return torch.from_numpy(center_prior.astype(np.float32))