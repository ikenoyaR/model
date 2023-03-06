import glob
import os
import random


import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.io import loadmat


def load_mat(path, dataset):
    if dataset == 'SALICON':
        img = np.zeros((1, 480, 640), dtype=np.uint8)
        fix_loc = loadmat(path)['gaze']['fixations'].reshape(-1)
        fix_loc = np.vstack([fix_loc[i] for i in range(len(fix_loc))])
        for w,h in fix_loc:
            img[0,h-1,w-1] = 1

    elif dataset == "CAT2000":
        img = np.array(loadmat(path)['fixLocs'], dtype=np.float32)

    elif dataset == 'MIT1003':
        img = cv2.imread(path)[:,:,0]
        img = np.where(img==0, 0, 1)
        img = img.astype(np.float32)
    elif dataset == 'DHF1K':
        img = loadmat(path)['I']
    else:
        raise NotImplementedError

    return img


class DHF1KDataset(Dataset):
    def __init__(self, path_data, len_snippet, img_height=224, img_width=384, mode="train", mode_sample='sliding_window', interval_sampling=1):
        '''
        mode: train, valid, test, predict
        input tensor, gt and fix are torch.tensor preprocessed and normalized 0 to 1.

        train, valid
            return video number(idx), image number, input tensor, ground truth
            Maps and images are sampled every videos.
        test
            return idx, input tensor, ground truth, fixations locations
            Maps, locs of fixations and images are sampled all frames of datasets.
            Video number and image number are got by using  DHF1KDataset._get_video_img_num() method.
        predict
            return idx, input tensor, images of input tensor[0]
            Images are sampled all frames of datasets.
            Video number and image number are got by using  DHF1KDataset._get_video_img_num() method.
        '''
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.mode_sample = mode_sample
        self.interval_sampling = interval_sampling
        self.list_sampler = self._make_sampler_list(self.len_snippet, self.mode_sample, self.interval_sampling)

        self.list_path_video = sorted(glob.glob(os.path.join(path_data, '*')))
        if self.mode == 'train':
            print('train data loading...')
            self.list_path_img = [sorted(glob.glob(os.path.join(path_video, 'images', '*.png'))) for path_video in tqdm(self.list_path_video)]
            self.list_path_map = [sorted(glob.glob(os.path.join(path_video, 'maps', '*.png'))) for path_video in tqdm(self.list_path_video)]
        elif self.mode == 'valid':
            print('validation data loading...')
            self.list_path_img = [sorted(glob.glob(os.path.join(path_video, 'images', '*.png'))) for path_video in tqdm(self.list_path_video)]
            self.list_path_map = [sorted(glob.glob(os.path.join(path_video, 'maps', '*.png'))) for path_video in tqdm(self.list_path_video)]
        elif self.mode == 'test':
            print('test data loading...')
            self.list_path_img = [sorted(glob.glob(os.path.join(path_video, 'images', '*.png'))) for path_video in tqdm(self.list_path_video)]
            self.list_path_map = [sorted(glob.glob(os.path.join(path_video, 'maps', '*.png'))) for path_video in tqdm(self.list_path_video)]
            self.list_path_fix = [sorted(glob.glob(os.path.join(path_video, 'fixation', '*.png'))) for path_video in tqdm(self.list_path_video)]
        elif self.mode == 'predict':
            print('data loading...')
            self.list_path_img = [sorted(glob.glob(os.path.join(path_video, 'images', '*.png'))) for path_video in tqdm(self.list_path_video)]
        else:
            raise NotImplementedError
        
        # self.list_num_img[idx] return video index and img_index
        self.list_num_img = [[video_num, img_num] for video_num, list_path_video in enumerate(self.list_path_img) for img_num, _ in enumerate(list_path_video)]
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            # transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        self.map_transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width))
        ])

        self.tensor_input = torch.zeros(3, self.len_snippet, self.img_height, self.img_width)

    def __len__(self):
        if self.mode == 'train':
            return len(self.list_path_img)
        elif self.mode == 'valid':
            return len(self.list_path_img)
        elif self.mode == 'test':
            return sum([len(list_path_img) for list_path_img in self.list_path_img])
        elif self.mode == 'predict':
            return sum([len(list_path_img) for list_path_img in self.list_path_img])
        else:
            raise NotImplementedError


    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'valid':
            self.tensor_input = torch.zeros(self.len_snippet, 3, self.img_height, self.img_width)
            video_num = idx
            img_num = np.random.randint(len(self.list_path_img[idx]))

            list_path_input_img_idx = [ img_num - num_sample_idx for num_sample_idx in self.list_sampler]
            idx_list_pop = None
            for idx_list, idx_path_input_img in enumerate(list_path_input_img_idx):
                if idx_path_input_img >= 0:
                    pass
                else:
                    idx_list_pop = idx_list
                    break
            if idx_list_pop != None:
                list_path_input_img_idx = list_path_input_img_idx[:idx_list_pop]

            for idx_tensor, img_idx in enumerate(list_path_input_img_idx):
                _img = self._get_img(video_num, img_idx, mode='img')
                _img = self._transform_img(_img)
                self.tensor_input[idx_tensor] = _img.clone()
            self.tensor_input = self.tensor_input.permute((1, 0, 2, 3))

            gt = self._get_img(video_num, img_num, mode='map')
            gt = torch.from_numpy(gt.astype(np.float32)).clone()
            gt = gt / 255
            assert self.list_path_img[video_num][img_num].split(os.sep)[-3] == self.list_path_map[video_num][img_num].split(os.sep)[-3], 'the video name of img and map are not same.'
            assert self.list_path_img[video_num][img_num].split(os.sep)[-1] == self.list_path_map[video_num][img_num].split(os.sep)[-1], 'the frame name of img and map are not same.'

            return video_num, img_num, self.tensor_input, gt
        elif self.mode == 'test' or self.mode == 'predict':
            self.tensor_input = torch.zeros(self.len_snippet, 3, self.img_height, self.img_width)
            video_num, _ = self._get_video_img_num(idx)
            list_path_input_img_idx = [ idx - num_sample_idx for num_sample_idx in self.list_sampler]
            idx_list_pop = None
            for idx_list, idx_path_input_img in enumerate(list_path_input_img_idx):
                if video_num == self.list_num_img[idx_path_input_img][0] and idx_path_input_img >= 0:
                    pass
                else:
                    idx_list_pop = idx_list
                    break
            if idx_list_pop != None:
                list_path_input_img_idx = list_path_input_img_idx[:idx_list_pop]

            for idx_tensor, idx_path_input_img in enumerate(list_path_input_img_idx):
                video_num, img_num = self._get_video_img_num(idx_path_input_img)
                _img = self._get_img(video_num, img_num, mode='img')
                _img = self._transform_img(_img)
                self.tensor_input[idx_tensor] = _img.clone()
            self.tensor_input = self.tensor_input.permute((1, 0, 2, 3))
            video_num, img_num = self._get_video_img_num(idx)
            if self.mode == 'predict':
                path_img = self.list_path_img[video_num][img_num]
                return idx, self.tensor_input, cv2.imread(path_img, 1)
            elif self.mode == 'test':
                gt = self._get_img(video_num, img_num, mode='map')
                gt = torch.from_numpy(gt.astype(np.float32)).clone()
                gt = gt / 255
                fixations = self._get_img(video_num, img_num, mode='fix')
                fixations = torch.from_numpy(np.where(fixations, 1, 0)).clone()
                assert self.list_path_img[video_num][img_num].split(os.sep)[-3] == self.list_path_map[video_num][img_num].split(os.sep)[-3], 'the video name of img and map are not same.'
                assert self.list_path_img[video_num][img_num].split(os.sep)[-1] == self.list_path_map[video_num][img_num].split(os.sep)[-1], 'the frame name of img and map are not same.'
                return idx, self.tensor_input, gt, fixations
        else:
            raise NotImplementedError


    def get_name_videoandimage(self, idx):
        if self.mode == 'predict':
            video_num, img_num = self._get_video_img_num(idx)
            video_name = self.list_path_img[video_num][img_num].split(os.sep)[-3]
            image_name = self.list_path_img[video_num][img_num].split(os.sep)[-1]
        else:
            raise NotImplementedError
        return video_name, image_name


    def _get_img(self, video_num, img_num, mode='img'):
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
            path_img = self.list_path_img[video_num][img_num]
            img = cv2.imread(path_img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'map':
            path_img = self.list_path_map[video_num][img_num]
            img = cv2.imread(path_img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif mode == 'fix':
            path_img = self.list_path_fix[video_num][img_num]
            img = cv2.imread(path_img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise NotImplementedError
        return img
    
    def _get_video_img_num(self, idx):
        return self.list_num_img[idx] # [video_num, img_num]

    def _make_sampler_list(self, len_temporal, mode_sample, interval_sampling):
        """
        mode_sample has not implemented
        mode_sample will sample close times densely and far times sparsely in the future.
        """
        if mode_sample == 'sliding_window':
            return [index*interval_sampling for index in range(len_temporal)]
        else:
            raise NotImplementedError
    
    def _transform_img(self, img):
        """
        input : ndarray(RGB image) (shape:(height, width, channel))
        output: torch.tensor (shape:(channel, height, width))
        """
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32)).clone()
        img = img / 255
        return self.img_transform(img)


class Sample_sauc_other_map(object):
    def __init__(self, path_data, n_sample=10, shape=(360, 640)):
        self.path_data = path_data
        self.n_sample = n_sample
        self.shape = shape
        self.list_path_fix = sorted(glob.glob(os.path.join(self.path_data, '*', 'fixation', '*.png')))
    
    def _get_img_fix(self, path_img):
        """(return C, H, W)"""
        img = cv2.imread(path_img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int32)
        return img
    
    def _get_path_img(self):
        list_path_fix = random.sample(self.list_path_fix, self.n_sample)
        return list_path_fix
    
    def get_sauc_other_map(self):
        list_path_fix = self._get_path_img()
        other_map = torch.zeros(self.shape[0], self.shape[1])
        for path_fix in list_path_fix:
            fix = self._get_img_fix(path_fix)
            other_map = torch.from_numpy(np.where(other_map + fix, 1, 0)).clone()
        return other_map


if __name__ == '__main__':
    train_data = DHF1KDataset(path_data = 'D:/ikenoya/datasets/DHF1K/train', len_snippet=16, mode='train')
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    valid_data = DHF1KDataset(path_data = 'D:/ikenoya/datasets/DHF1K/val', len_snippet=16, mode='valid')
    valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False)
    for video_num, img_num, input, gt in train_dataloader:
        print(video_num, img_num, input.shape, gt.shape)
    
    for video_num, img_num, input, gt in valid_dataloader:
        print(video_num, img_num, input.shape, gt.shape)
