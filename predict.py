import os
from collections import OrderedDict


import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import torchvision
torch.manual_seed(0)


from datasets.dhf1k import DHF1KDataset
from utils.loss import Postprocess
from utils.utils import normalize_map


def predict(args, model):
    """
    generate predicted saliency model
    """
    test_data = DHF1KDataset(path_data=os.path.join(args.root, 'test'), len_snippet=args.temporal_length, mode='predict',
                             mode_sample=args.mode_sampling_input, interval_sampling=args.interval_sampling_input)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.pin_memory, shuffle=False)
    
    post_process = Postprocess(output_shape=(360, 640))

    model.eval()
    with torch.no_grad():
        with tqdm(test_dataloader) as pbar:
            for idx, inputs, img in pbar:
                inputs = inputs.to(args.device)
    
                pred = model(inputs)
                pred = normalize_map(pred, regularize=True)
                pred = post_process.transform_predicted_map(pred, blur=True)
                pred = torch.round(pred.mul(255).add(0.5).clamp(0,255)).to('cpu', torch.uint8).detach().numpy().copy()

                for b in range(pred.shape[0]):
                    video_name, image_name = test_data.get_name_videoandimage(idx[b])
                    path_video = os.path.join(args.save_path, video_name)
                    if not os.path.exists(path_video):
                        os.makedirs(path_video)
                    path_image = os.path.join(path_video, image_name)
                    if not args.nosave:
                        cv2.imwrite(path_image, pred[b])
                    
                    if args.nosave or args.visualize or args.save_heatmap:
                        color_pred = cv2.applyColorMap(pred[b], cv2.COLORMAP_JET)
                        image = img[b].to('cpu').detach().numpy().copy()
                        image = cv2.resize(image, dsize=(color_pred.shape[1], color_pred.shape[0]))
                        combined_img = cv2.addWeighted(src1=image,
                                                       alpha=0.9, src2=color_pred, beta=0.55, gamma=0)
                        if args.nosave or args.visualize:
                            cv2.imshow('saliency map', combined_img)
                            cv2.waitKey(1)
                        if args.save_heatmap:
                            if not os.path.exists(os.path.join(path_video, 'overlay')):
                                os.makedirs(os.path.join(path_video, 'overlay'))
                            path_combined_img = os.path.join(path_video, 'overlay', image_name)
                            cv2.imwrite(path_combined_img, combined_img)
                pbar.set_postfix(OrderedDict(video=f'{test_data.list_num_img[idx[0]][0]:04}', image=f'{test_data.list_num_img[idx[0]][1]:04}'))
                
    cv2.destroyAllWindows()
    
    

