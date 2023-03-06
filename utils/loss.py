

import torch
import torch.nn as nn
from torchvision import transforms


from .eval_metric import kl_func, cc_func, sim_func, ig_func
from utils.utils import get_ig_baseline

class Postprocess(object):
    def __init__(self, output_shape=(360, 640), mode='bilinear', align_corners=False, gaussian_kernel=11):
        self.output_shape = output_shape
        self.upsample = nn.Upsample(size=output_shape, mode=mode, align_corners=align_corners)
        _sigma = 0.3*((gaussian_kernel-1)*0.5 - 1) + 0.8 # from opencv gaussianblur
        # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=gaussian_kernel, sigma=_sigma)

    def transform_predicted_map(self, pred_map, blur=False):
        pred_map = pred_map.unsqueeze(1)
        pred_map = self.upsample(pred_map)
        pred_map = pred_map.squeeze(1)
        if blur:
            pred_map = self.gaussian_blur(pred_map)
        return pred_map

    def confirm_shape(self, gt_map):
        w = gt_map.shape[-1]
        h = gt_map.shape[-2]
        return (h == self.output_shape[0]) and (w == self.output_shape[1])


class Lossmanager(object):
    def __init__(self, args):
        self.args = args
        if self.args.ig:
            self.ig_baseline = get_ig_baseline(args.root_ig_baseline).to(self.args.device)

    def get_loss(self, pred_map, gt):
        loss = torch.FloatTensor([0.0]).to(self.args.device)
        if self.args.kl:
            loss += self.args.kl_coeff * kl_func(pred_map, gt)
        if self.args.cc:
            loss += self.args.cc_coeff * cc_func(pred_map, gt)
        if self.args.sim:
            loss += self.args.sim_coeff * sim_func(pred_map, gt)
        if self.args.ig:
            loss += self.args.ig_coeff * ig_func(pred_map, gt, self.ig_baseline)
    
        return loss
    
    def return_loss(self, pred_map, gt):
        """loss (averaged in minibatch)"""
        loss = torch.FloatTensor([0.0]).to(self.args.device)
        assert pred_map.size() == gt.size()
    
        if len(pred_map.size()) == 4:
            ''' Clips: BxClXHxW '''
            assert pred_map.size(0) == self.args.batch_size
            pred_map = pred_map.permute((1,0,2,3))
            gt = gt.permute((1,0,2,3))
    
            for i in range(pred_map.size(0)):
                loss += self.get_loss(pred_map[i], gt[i])
    
            loss /= pred_map.size(0)
            return loss
        
        return self.get_loss(pred_map, gt)

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, sample = 1):
        self.val = val
        self.sum += val * sample
        self.count += sample
        self.avg = self.sum / self.count