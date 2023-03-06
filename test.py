import os
import csv
import sys
from collections import OrderedDict


from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import torchvision
torch.manual_seed(0)


from datasets.dhf1k import DHF1KDataset, Sample_sauc_other_map
from utils.evaluate import Evaluate_saliency
from utils.logger import write_csv, write_test_info
from utils.loss import Postprocess, AverageMeter
from utils.utils import normalize, get_ig_baseline


def test(args, model):
    """
    test model
    
    future work : output value of evaluation every video
    (video number and image number are needed)
    """
    write_test_info(os.path.join(args.save_path, 'test_info.csv'), args)
    print('evaluating by validation data')
    test_data = DHF1KDataset(path_data=os.path.join(args.root, 'val'), len_snippet=args.temporal_length, mode='test',
                              mode_sample=args.mode_sampling_input, interval_sampling=args.interval_sampling_input)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.pin_memory, shuffle=False)
    
    
    eval = Evaluate_saliency(args=args, gt_map=True, gt_loc=True,
                             sauc_other_map=os.path.exists(args.root_sauc_other_map),
                             center_prior=os.path.exists(args.root_ig_baseline))
    ig_baseline = get_ig_baseline(args.root_ig_baseline)
    print('evaluate metric : ',', '.join(eval.list_metric))
    sampler_other_map = Sample_sauc_other_map(args.root_sauc_other_map)
                             
    metric_avg_managers = { metric:AverageMeter() for metric in eval.list_metric} # 'kl','cc','sim', etc...

    post_process = Postprocess(output_shape=(360, 640))

    # write_csv(os.path.join(args.save_path, 'loss.csv'), (['epoch', 'train_loss', 'val_loss']), mode='w')

    model.eval()
    with torch.no_grad():
        with tqdm(test_dataloader) as pbar:
            for idx, inputs, gt_map, gt_loc in pbar:
                inputs = inputs.to(args.device)
                gt_map = gt_map.to(args.device)
                gt_loc = gt_loc.to(args.device)
    
                pred = model(inputs)
                pred = post_process.transform_predicted_map(pred, blur=True)
                assert post_process.confirm_shape(gt_map), 'predicted map and ground truth map are not same shape'
                
                if args.visualize:
                    torchvision.utils.save_image(inputs.permute(0,2,1,3,4)[0], os.path.join(args.save_path, 'input.png'))
                    torchvision.utils.save_image(gt_map[0], os.path.join(args.save_path, 'gt_map.png'))
                    # torchvision.utils.save_image(gt_loc[0], os.path.join(args.save_path, 'gt_loc.png'))
                    torchvision.utils.save_image(normalize(pred[0]), os.path.join(args.save_path, 'output.png'))
                
                sauc_other_map = sampler_other_map.get_sauc_other_map()
                if args.visualize:
                    torchvision.utils.save_image(sauc_other_map.to(torch.float), os.path.join(args.save_path, 'other_map.png'))

                eval_metrics = eval.eval_map(pred, gt_map, gt_loc, sauc_other_map, ig_baseline)
                for metric in eval.list_metric:
                    metric_avg_managers[metric].update(eval_metrics[metric], sample=args.batch_size)
                pbar.set_postfix(OrderedDict(video=f'{test_data.list_num_img[idx[0]][0]:04}', image=f'{test_data.list_num_img[idx[0]][1]:04}'))
                
    
                # pbar.set_postfix(metric_avg_managers)
    write_csv(os.path.join(args.save_path, 'eval.csv'), eval.list_metric, mode='w')
    write_csv(os.path.join(args.save_path, 'eval.csv'), [metric_avg_managers[metric].avg for metric in eval.list_metric])

    

