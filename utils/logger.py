import csv
from typing import NoReturn
import argparse


from . import *

def write_csv(csv_path, row_write, mode='a'):
    """mode choices:['w', 'a']"""
    with open(csv_path, mode, newline='') as f:
        w = csv.writer(f)
        w.writerow(row_write)


def write_txt(txt_path, list_write, delimiter=', ', mode='a'):
    """mode choices:['w', 'a']"""
    list_write = [str(elements) for elements in list_write]
    list_write = delimiter.join(list_write)
    with open(txt_path, mode=mode, encoding='utf-8') as f:
        f.writelines(list_write)


def log_info_csv(csv_path:str, contents_list) -> NoReturn:
    for i, contents in enumerate(contents_list):
        if i == 0:
            write_csv(csv_path, contents, mode='w')
        else:
            write_csv(csv_path, contents, mode='a')


def log_info_txt(txt_path:str, contents_list) -> NoReturn:
    for i, contents in enumerate(contents_list):
        if i == 0:
            write_txt(txt_path, list(contents) + ['\n'], mode='w')
        else:
            write_txt(txt_path, list(contents) + ['\n'], mode='a')


def args2dict(args:argparse.Namespace) -> dict:
    return vars(args)


def log_train_info(txt_path:str, args:argparse.Namespace) -> NoReturn:
    dict_args = args2dict(args)
    log_info_txt(txt_path, list(dict_args.items()))
    

def write_train_info(path_csv:str, args:argparse.Namespace):
    dict_args = args2dict(args)
    write_csv(path_csv, ['train_info'], mode='w')
    for content in list(dict_args.items()):
        write_csv(path_csv, content, mode='a')

def write_test_info(path_csv:str, args:argparse.Namespace):
    write_csv(path_csv, ['test info'], mode='w')
    write_csv(path_csv, ['model', args.model])
    write_csv(path_csv, ['path load weight', args.path_load_weight])
    write_csv(path_csv, ['input shape', f'3x{args.temporal_length}x{args.image_height}x{args.image_width}'])
    write_csv(path_csv, ['sampling', args.mode_sampling_input, args.interval_sampling_input])

