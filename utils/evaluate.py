import os


import torch
import numpy as np
import cv2


class Evaluate_saliency(object):
    """
    return dict{'kl': , 'cc': , 'sim': ,
                'auc': , 'nss': , 'sauc':,
                'ig':}
    """
    def __init__(self, args, gt_map=True, gt_loc=True, sauc_other_map=True, center_prior=True):
        self.device = args.device
        self.gt_map = gt_map
        self.gt_loc = gt_loc
        self.sauc_other_map = sauc_other_map
        self.center_prior = center_prior
        self.list_metric = self.decide_metric()
    
    def decide_metric(self):
        list_metric = []
        if self.gt_map:
            list_metric.append('kl')
            list_metric.append('cc')
            list_metric.append('sim')
        if self.gt_loc:
            list_metric.append('nss')
            list_metric.append('auc')
            if self.sauc_other_map:
                list_metric.append('sauc')
            if self.center_prior:
                list_metric.append('ig')
        return list_metric

    def eval_map(self, s_map, gt_map=None, gt_loc=None, sauc_other_map=None, center_prior=None):
        value_metric = {}
        if self.gt_map:
            value_metric['kl'] = self.kl_func(s_map.to(self.device), gt_map.to(self.device)).item()
            value_metric['cc'] = self.cc_func(s_map.to(self.device), gt_map.to(self.device)).item()
            value_metric['sim'] = self.sim_func(s_map.to(self.device), gt_map.to(self.device)).item()
        if self.gt_loc:
            value_metric['auc'] = float(self.auc_judd_func(s_map.to('cpu'), gt_loc.to('cpu'), jitter=True, normalize=False))
            value_metric['nss'] = self.nss_func(s_map.to(self.device), gt_loc.to(self.device)).item()
            if self.sauc_other_map:
                # value_metric['sauc'] = float(self.auc_shuff_func(s_map.to('cpu'), gt_loc.to('cpu'), sauc_other_map))
                value_metric['sauc'] = float(self.sauc_func(s_map.to('cpu'), gt_loc.to('cpu'), sauc_other_map))
            if self.center_prior:
                value_metric['ig'] = self.ig_func(s_map.to(self.device), gt_loc.to(self.device), center_prior.to(self.device)).item()
        return value_metric

    def kl_func(self, s_map, gt):
        assert s_map.size() == gt.size()
        batch_size = s_map.size(0)
        h = s_map.size(1)
        w = s_map.size(2)

        sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
        expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, h, w)

        assert expand_s_map.size() == s_map.size()

        sum_gt = torch.sum(gt.view(batch_size, -1), 1)
        expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, h, w)

        assert expand_gt.size() == gt.size()

        s_map = s_map/(expand_s_map*1.0)
        gt = gt / (expand_gt*1.0)

        s_map = s_map.view(batch_size, -1)
        gt = gt.view(batch_size, -1)

        eps = 2.2204e-16
        result = gt * torch.log(eps + gt/(s_map + eps))
        # print(torch.log(eps + gt/(s_map + eps))   )
        return torch.mean(torch.sum(result, 1))


    def normalize_map(self, s_map):
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

        norm_s_map = (s_map - min_s_map)/(max_s_map-min_s_map*1.0)
        return norm_s_map


    def sim_func(self, s_map, gt):
        ''' For single image metric
            Size of Image - WxH or 1xWxH
            gt is ground truth saliency map
        '''
        batch_size = s_map.size(0)
        h = s_map.size(1)
        w = s_map.size(2)

        s_map = self.normalize_map(s_map)
        gt = self.normalize_map(gt)

        sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
        expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, h, w)

        assert expand_s_map.size() == s_map.size()

        sum_gt = torch.sum(gt.view(batch_size, -1), 1)
        expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, h, w)

        s_map = s_map/(expand_s_map*1.0)
        gt = gt / (expand_gt*1.0)

        s_map = s_map.view(batch_size, -1)
        gt = gt.view(batch_size, -1)
        return torch.mean(torch.sum(torch.min(s_map, gt), 1))


    def cc_func(self, s_map, gt):
        assert s_map.size() == gt.size()
        batch_size = s_map.size(0)
        h = s_map.size(1)
        w = s_map.size(2)

        mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, h, w)
        std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, h, w)

        mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, h, w)
        std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, h, w)

        s_map = (s_map - mean_s_map) / std_s_map
        gt = (gt - mean_gt) / std_gt

        ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
        aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
        bb = torch.sum((gt * gt).view(batch_size, -1), 1)

        return torch.mean(ab / (torch.sqrt(aa*bb)))


    def nss_func(self, s_map, gt):
        if s_map.size() != gt.size():
            s_map = s_map.cpu().squeeze(0).numpy()
            s_map = torch.FloatTensor(cv2.resize(s_map, (gt.size(2), gt.size(1)))).unsqueeze(0)
            s_map = s_map.cuda()
            gt = gt.cuda()
        # print(s_map.size(), gt.size())
        assert s_map.size()==gt.size()
        batch_size = s_map.size(0)
        h = s_map.size(1)
        w = s_map.size(2)
        mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, h, w)
        std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, h, w)

        eps = 2.2204e-16
        s_map = (s_map - mean_s_map) / (std_s_map + eps)

        s_map = torch.sum((s_map * gt).view(batch_size, -1), 1)
        count = torch.sum(gt.view(batch_size, -1), 1)
        return torch.mean(s_map / count)


    def auc_judd_func(self, saliencyMap, fixationMap, jitter=True, toPlot=False, normalize=False):
        """
        saliencyMap is the saliency map
        fixationMap is the human fixation map (binary matrix)
        jitter=True will add tiny non-zero random constant to all map locations to ensure
              ROC can be calculated robustly (to avoid uniform region)
        if toPlot=True, displays ROC curve

        If there are no fixations to predict, return NaN
        """
        if saliencyMap.size() != fixationMap.size():
            saliencyMap = saliencyMap.cpu().squeeze(0).numpy()
            saliencyMap = torch.FloatTensor(cv2.resize(saliencyMap, (fixationMap.size(2), fixationMap.size(1)))).unsqueeze(0)
            # saliencyMap = saliencyMap.cuda()
            # fixationMap = fixationMap.cuda()
        if len(saliencyMap.size())==3:
            saliencyMap = saliencyMap[0,:,:]
            fixationMap = fixationMap[0,:,:]
        saliencyMap = saliencyMap.detach().numpy()
        fixationMap = fixationMap.detach().numpy()
        if normalize:
            saliencyMap = self.normalize_map(saliencyMap)

        if not fixationMap.any():
            print('Error: no fixationMap')
            score = float('nan')
            return score

        # make the saliencyMap the size of the image of fixationMap

        if not np.shape(saliencyMap) == np.shape(fixationMap):
            from scipy.misc import imresize
            saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

        # jitter saliency maps that come from saliency models that have a lot of zero values.
        # If the saliency map is made with a Gaussian then it does not need to be jittered as
        # the values are varied and there is not a large patch of the same value. In fact
        # jittering breaks the ordering in the small values!
        if jitter:
            # jitter the saliency map slightly to distrupt ties of the same numbers
            saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

        # normalize saliency map
        saliencyMap = (saliencyMap - saliencyMap.min()) \
                      / (saliencyMap.max() - saliencyMap.min())

        if np.isnan(saliencyMap).all():
            print('NaN saliencyMap')
            score = float('nan')
            return score

        S = saliencyMap.flatten()
        F = fixationMap.flatten()

        Sth = S[F > 0]  # sal map values at fixation locations
        Nfixations = len(Sth)
        Npixels = len(S)

        allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
        tp = np.zeros((Nfixations + 2))
        fp = np.zeros((Nfixations + 2))
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i in range(Nfixations):
            thresh = allthreshes[i]
            aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
            tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
            # above threshold
            fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
            # above threshold

        score = np.trapz(tp, x=fp)
        allthreshes = np.insert(allthreshes, 0, 0)
        allthreshes = np.append(allthreshes, 1)

        if toPlot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.matshow(saliencyMap, cmap='gray')
            ax.set_title('SaliencyMap with fixations to be predicted')
            [y, x] = np.nonzero(fixationMap)
            s = np.shape(saliencyMap)
            plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
            plt.plot(x, y, 'ro')

            ax = fig.add_subplot(1, 2, 2)
            plt.plot(fp, tp, '.b-')
            ax.set_title('Area under ROC curve: ' + str(score))
            plt.axis((0, 1, 0, 1))
            plt.show()

        return score


    def sauc_func(self, s_map, gt, other_map, n_splits=100, stepsize=0.1):
        """
        s_map.shape is (B, H, W) sauc pytorch implementation
        ref : https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py
        """
        s_map = self.normalize_map(s_map)
        s_map = s_map.view(s_map.shape[0], -1)
        gt = gt.view(gt.shape[0], -1)
        # other_map = other_map.expand(s_map.shape[0], -1, -1).view(s_map.shape[0], -1)
        other_map= other_map.view(-1)

        list_s_map_th = [s_map[i][gt[i] > 0] for i in range(s_map.shape[0])]   # sal map values at fixation locations
        list_n_fixations = [s_map_th.shape[0] for s_map_th in list_s_map_th]

        ind = torch.where(other_map > 0)[0]

        list_n_fixations_other_map = [min(n_fixations, len(ind)) for n_fixations in list_n_fixations]
        list_randfix = [torch.full((n_fixations_other_map, n_splits), float('nan')) for n_fixations_other_map in list_n_fixations_other_map]
        
        for b, n_fixations_other_map in enumerate(list_n_fixations_other_map):
            for i in range(n_splits):
                # randomize choice of fixation locations
                randind = torch.tensor(np.random.permutation(ind.clone()))
                # sal map values at random fixation locations of other random images
                list_randfix[b][:, i] = s_map[b, randind[:n_fixations_other_map]]

        list_auc = torch.full((len(list_randfix), n_splits), float('nan'))
        for b, randfix in enumerate(list_randfix):
            for s in range(n_splits):
                curfix = randfix[:, s]
                allthreshes = torch.flip(torch.arange(0, max(list_s_map_th[b].max(), curfix.max()), stepsize), dims=[0])
                tp = torch.zeros(len(allthreshes) + 2)
                fp = torch.zeros(len(allthreshes) + 2)
                tp[-1] = 1
                fp[-1] = 1
                
                for i, thresh in enumerate(allthreshes):
                    tp[i + 1] = torch.sum(list_s_map_th[b] >= thresh)/list_n_fixations[b]
                    fp[i + 1] = torch.sum(curfix >= thresh) / list_n_fixations_other_map[b]
                list_auc[b, s] = torch.trapz(tp, fp)
        return torch.mean(list_auc)


    def ig_func(self, s_map, gt, baseline_map):
        assert s_map.size() == gt.size()
        assert s_map.size(-2) == baseline_map.size(0)
        assert s_map.size(-1) == baseline_map.size(1)
        batch_size = s_map.size(0)
        h = s_map.size(1)
        w = s_map.size(2)

        sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
        expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, h, w)

        assert expand_s_map.size() == s_map.size()


        sum_baseline_map = torch.sum(baseline_map)
        expand_baseline_map = sum_baseline_map


        s_map = s_map/(expand_s_map*1.0)
        baseline_map = baseline_map / (expand_baseline_map*1.0)

        s_map = s_map.view(batch_size, -1)
        baseline_map = baseline_map.view(-1).expand_as(s_map) # equal to : baseline_map.view(-1).unsqueeze(0).expand(batch_size, -1)
        gt = gt.view(batch_size, -1)
        num_gt_fix_loc = (gt.shape[-1]-(gt == 0).sum(dim=1))
        assert num_gt_fix_loc.nonzero().shape[0] == num_gt_fix_loc.shape[0], 'ground truth(fixation location) include zero data.'

        eps = 2.2204e-16
        result = gt * (torch.log2( eps + s_map )-torch.log2( eps + baseline_map ))
        # print(torch.log(eps + gt/(s_map + eps))   )
        return torch.mean(torch.sum(result, 1) / (num_gt_fix_loc + eps))
    
