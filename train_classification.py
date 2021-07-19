import logging
from typing import List

import torch
# import torch should be first. Unclear issue, mentioned here: https://github.com/pytorch/pytorch/issues/2083
import argparse
import os
from pathlib import Path
import csv
import time
import h5py
from _datetime import datetime
import warnings

from tqdm import tqdm
from collections import OrderedDict
import shutil

try:
    from pynvml import *
except ModuleNotFoundError:
    warnings.warn(f"The python Nvidia management library could not be imported. Ignore if running on CPU only.")

import torchvision
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from utils.optimizer import create_optimizer
from utils.logger import InformationLogger, save_logs_to_bucket, tsv_line
from utils.metrics import report_classification, create_metrics_dict
from models.model_choice import net, load_checkpoint
from losses import MultiClassCriterion
from utils.utils import load_from_checkpoint, list_s3_subfolders, get_device_ids, gpu_stats, \
    get_key_def
from utils.readers import read_parameters

try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass

logging.getLogger(__name__)


def verify_weights(num_classes, weights):
    """Verifies that the number of weights equals the number of classes if any are given
    Args:
        num_classes: number of classes defined in the configuration file
        weights: weights defined in the configuration file
    """
    if num_classes == 1 and len(weights) == 2:
        warnings.warn("got two class weights for single class defined in configuration file; will assume index 0 = background")
    elif num_classes != len(weights):
        raise ValueError('The number of class weights in the configuration file is different than the number of classes')


def flatten_labels(annotations):
    """Flatten labels"""
    flatten = annotations.view(-1)
    return flatten


def flatten_outputs(predictions, number_of_classes):
    """Flatten the prediction batch except the prediction dimensions"""
    logits_permuted = predictions.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    outputs_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return outputs_flatten


def loader(path):
    img = Image.open(path)
    return img


def get_s3_classification_images(dataset, bucket, bucket_name, data_path, output_path, num_classes):
    classes = list_s3_subfolders(bucket_name, os.path.join(data_path, dataset))
    classes.sort()
    assert num_classes == len(classes), "The configuration file specified %d classes, but only %d class folders were " \
                                        "found in %s." % (num_classes, len(classes), os.path.join(data_path, dataset))
    with open(os.path.join(output_path, 'classes.csv'), 'wt') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(classes)

    path = os.path.join('Images', dataset)
    try:
        os.mkdir(path)  # TODO use Path from pathlib instead?
    except FileExistsError:
        pass
    for c in classes:
        classpath = os.path.join(path, c)
        try:
            os.mkdir(classpath)
        except FileExistsError:
            pass
        for f in bucket.objects.filter(Prefix=os.path.join(data_path, dataset, c)):
            if f.key != data_path + '/':
                bucket.download_file(f.key, os.path.join(classpath, f.key.split('/')[-1]))


def get_local_classes(num_classes, data_path, output_path):
    # Get classes locally and write to csv in output_path
    classes = next(os.walk(os.path.join(data_path, 'trn')))[1]
    classes.sort()
    assert num_classes == len(classes), "The configuration file specified %d classes, but only %d class folders were " \
                                        "found in %s." % (num_classes, len(classes), os.path.join(data_path, 'trn'))
    with open(os.path.join(output_path, 'classes.csv'), 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(classes)


def download_s3_files(bucket_name, data_path, output_path, num_classes):
    """
    Function to download the required training files from s3 bucket and sets ec2 paths.
    :param bucket_name: (str) bucket in which data is stored if using AWS S3
    :param data_path: (str) EC2 file path of the folder containing h5py files
    :param output_path: (str) EC2 file path in which the model will be saved
    :param num_classes: (int) number of classes
    :return: (S3 object) bucket, (str) bucket_output_path, (str) local_output_path, (str) data_path
    """
    bucket_output_path = output_path
    local_output_path = 'output_path'
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    for i in ['trn', 'val', 'tst']:
        get_s3_classification_images(i, bucket, bucket_name, data_path, output_path, num_classes)
        class_file = os.path.join(output_path, 'classes.csv')
        bucket.upload_file(class_file, os.path.join(bucket_output_path, 'classes.csv'))
    data_path = 'Images'

    return bucket, bucket_output_path, local_output_path, data_path


def create_classif_dataloader(data_path, batch_size, num_devices):
    """
    Function to create dataloader objects for training, validation and test datasets.
    :param data_path: (str) path to the samples folder
    :param batch_size: (int) batch size
    :param num_devices: (int) number of GPUs used
    :return: trn_dataloader, val_dataloader, tst_dataloader
    """
    num_samples = {}
    trn_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, "trn"),
                                                   transform=transforms.Compose(
                                                       [transforms.RandomRotation((0, 275)),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.Resize(299), transforms.ToTensor()]),
                                                   loader=loader)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, "val"),
                                                   transform=transforms.Compose(
                                                       [transforms.Resize(299), transforms.ToTensor()]),
                                                   loader=loader)
    tst_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, "tst"),
                                                   transform=transforms.Compose(
                                                       [transforms.Resize(299), transforms.ToTensor()]),
                                                   loader=loader)
    # FIXME assert that f is a file
    num_samples['tst'] = len([f for f in Path(data_path).joinpath('tst').glob('**/*')])

    # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    num_workers = num_devices * 4 if num_devices > 1 else 4

    # Shuffle must be set to True.
    trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    tst_dataloader = DataLoader(tst_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False) if num_samples['tst'] > 0 else None

    return trn_dataloader, val_dataloader, tst_dataloader


def get_num_samples(samples_path, params):
    """
    Function to retrieve number of samples, either from config file or directly from hdf5 file.
    :param samples_path: (str) Path to samples folder
    :param params: (dict) Parameters found in the yaml config file.
    :return: (dict) number of samples for trn, val and tst.
    """
    num_samples = {'trn': 0, 'val': 0, 'tst': 0}

    for i in ['trn', 'val', 'tst']:
        if params['training'][f"num_{i}_samples"]:
            num_samples[i] = params['training'][f"num_{i}_samples"]

            with h5py.File(os.path.join(samples_path, f"{i}_samples.hdf5"), 'r') as hdf5_file:
                file_num_samples = len(hdf5_file['map_img'])
            if num_samples[i] > file_num_samples:
                raise IndexError(f"The number of training samples in the configuration file ({num_samples[i]}) "
                                 f"exceeds the number of samples in the hdf5 training dataset ({file_num_samples}).")
        else:
            with h5py.File(os.path.join(samples_path, f"{i}_samples.hdf5"), "r") as hdf5_file:
                num_samples[i] = len(hdf5_file['map_img'])

    return num_samples


def main(params, config_path):
    """
    Function to train and validate a models for semantic segmentation or classification.
    :param params: (dict) Parameters found in the yaml config file.
    :param config_path: (str) Path to the yaml config file.

    """
    # MANDATORY PARAMETERS
    num_classes = get_key_def('num_classes', params['global'], expected_type=int)
    num_bands = get_key_def('number_of_bands', params['global'], expected_type=int)
    batch_size = get_key_def('batch_size', params['training'], expected_type=int)
    num_epochs = get_key_def('num_epochs', params['training'], expected_type=int)
    model_name = get_key_def('model_name', params['global'], expected_type=str).lower()
    BGR_to_RGB = get_key_def('BGR_to_RGB', params['global'], expected_type=bool)

    # parameters to find hdf5 samples
    data_path = Path(get_key_def('data_path', params['global'], './data', expected_type=str))
    assert data_path.is_dir(), f'Could not locate data path {data_path}'

    # OPTIONAL PARAMETERS
    # basics
    debug = get_key_def('debug_mode', params['global'], default=False, expected_type=bool)
    task = get_key_def('task', params['global'], default='classification', expected_type=str)
    assert task == 'classification', f"The task should be classification. The provided value is {task}"
    dontcare_val = get_key_def("ignore_index", params["training"], default=-1, expected_type=int)
    batch_metrics = get_key_def('batch_metrics', params['training'], default=1, expected_type=int)
    meta_map = get_key_def("meta_map", params["global"], default={})
    bucket_name = get_key_def('bucket_name', params['global'])  # AWS

    # model params
    loss_fn = get_key_def('loss_fn', params['training'], default='CrossEntropy', expected_type=str)
    optimizer = get_key_def('optimizer', params['training'], default='adam', expected_type=str)
    pretrained = get_key_def('pretrained', params['training'], default=True, expected_type=bool)
    train_state_dict_path = get_key_def('state_dict_path', params['training'], default=None, expected_type=str)
    dropout_prob = get_key_def('dropout_prob', params['training'], default=None, expected_type=float)

    # gpu parameters
    num_devices = get_key_def('num_gpus', params['global'], default=0, expected_type=int)
    max_used_ram = get_key_def('max_used_ram', params['global'], default=2000, expected_type=int)
    max_used_perc = get_key_def('max_used_perc', params['global'], default=15, expected_type=int)

    # automatic model naming with unique id for each training
    model_id = config_path.stem
    output_path = data_path.joinpath('model') / model_id
    if output_path.is_dir():
        last_mod_time_suffix = datetime.fromtimestamp(output_path.stat().st_mtime).strftime('%Y%m%d-%H%M%S')
        archive_output_path = data_path.joinpath('model') / f"{model_id}_{last_mod_time_suffix}"
        shutil.move(output_path, archive_output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(str(config_path), str(output_path))  # copy yaml to output path where model will be saved
    tqdm.write(f'Model and log files will be saved to: {output_path}\n\n')

    if debug:
        warnings.warn(f'Debug mode activated. Some debug functions may cause delays in execution.')

    if bucket_name:
        bucket, bucket_output_path, output_path, data_path = download_s3_files(bucket_name=bucket_name,
                                                                               data_path=data_path,
                                                                               output_path=output_path,
                                                                               num_classes=num_classes)
    elif not bucket_name:
        get_local_classes(num_classes, data_path, output_path)

    since = time.time()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    best_loss = 999

    progress_log = Path(output_path) / 'progress.log'
    if not progress_log.exists():
        progress_log.open('w', buffering=1).write(tsv_line('ep_idx', 'phase', 'iter', 'i_p_ep', 'time'))  # Add header

    trn_log = InformationLogger('trn')
    val_log = InformationLogger('val')
    tst_log = InformationLogger('tst')

    # list of GPU devices that are available and unused. If no GPUs, returns empty list
    gpu_devices_dict = get_device_ids(num_devices,
                                      max_used_ram_perc=max_used_ram,
                                      max_used_perc=max_used_perc)
    num_devices = len(gpu_devices_dict.keys())
    device = torch.device(f'cuda:0' if gpu_devices_dict else 'cpu')

    tqdm.write(f'Creating dataloaders from data in {Path(data_path)}...\n')
    trn_dataloader, val_dataloader, tst_dataloader = create_classif_dataloader(data_path=data_path,
                                                                               batch_size=batch_size,
                                                                               num_devices=num_devices,)

    # INSTANTIATE MODEL AND LOAD CHECKPOINT FROM PATH
    model, model_name, criterion, optimizer, lr_scheduler = net(model_name=model_name,
                                                                num_bands=num_bands,
                                                                num_channels=num_classes,
                                                                dontcare_val=dontcare_val,
                                                                num_devices=num_devices,
                                                                train_state_dict_path=train_state_dict_path,
                                                                pretrained=pretrained,
                                                                dropout_prob=dropout_prob,
                                                                loss_fn=loss_fn,
                                                                optimizer=optimizer,
                                                                net_params=params)
    tqdm.write(f'Instantiated {model_name} model with {num_classes} output channels.\n')

    filename = os.path.join(output_path, 'checkpoint.pth.tar')

    for epoch in range(0, params['training']['num_epochs']):
        logging.info(f'\nEpoch {epoch}/{params["training"]["num_epochs"] - 1}\n{"-" * 20}')

        trn_report = train(train_loader=trn_dataloader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=lr_scheduler,
                           num_classes=num_classes,
                           batch_size=batch_size,
                           ep_idx=epoch,
                           progress_log=progress_log,
                           device=device,
                           debug=debug)
        trn_log.add_values(trn_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        val_report = evaluation(eval_loader=val_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                ep_idx=epoch,
                                progress_log=progress_log,
                                batch_metrics=params['training']['batch_metrics'],
                                dataset='val',
                                device=device,
                                debug=debug)
        val_loss = val_report['loss'].avg
        if params['training']['batch_metrics'] is not None:
            val_log.add_values(val_report, epoch, ignore=['iou'])
        else:
            val_log.add_values(val_report, epoch, ignore=['precision', 'recall', 'fscore', 'iou'])

        if val_loss < best_loss:
            tqdm.write("save checkpoint\n")
            best_loss = val_loss
            # More info: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
            state_dict = model.module.state_dict() if num_devices > 1 else model.state_dict()
            torch.save({'epoch': epoch,
                        'params': params,
                        'model': state_dict,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()}, filename)

            if bucket_name:
                bucket_filename = os.path.join(bucket_output_path, 'checkpoint.pth.tar')
                bucket.upload_file(filename, bucket_filename)

        if bucket_name:
            save_logs_to_bucket(bucket, bucket_output_path, output_path, batch_metrics)

        cur_elapsed = time.time() - since
        logging.info(f'Current elapsed time {cur_elapsed // 60:.0f}m {cur_elapsed % 60:.0f}s')

    # load checkpoint model and evaluate it on test dataset.
    if int(params['training']['num_epochs']) > 0:  # if num_epochs is set to 0, model is loaded to evaluate on test set
        checkpoint = load_checkpoint(filename)
        model, _ = load_from_checkpoint(checkpoint, model)

    if tst_dataloader:
        tst_report = evaluation(eval_loader=tst_dataloader,
                                model=model,
                                criterion=criterion,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                ep_idx=num_epochs,
                                progress_log=progress_log,
                                batch_metrics=batch_metrics,
                                dataset='tst',
                                device=device)
        tst_log.add_values(tst_report, num_epochs, ignore=['iou'])

        if bucket_name:
            bucket_filename = os.path.join(bucket_output_path, 'last_epoch.pth.tar')
            bucket.upload_file("output.txt", os.path.join(bucket_output_path, f"Logs/{now}_output.txt"))
            bucket.upload_file(filename, bucket_filename)

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train(train_loader, model, criterion, optimizer, scheduler, num_classes, batch_size, ep_idx, progress_log, device, debug=False):
    """
    Train the model and return the metrics of the training epoch
    :param train_loader: training data loader
    :param model: model to train
    :param criterion: loss criterion
    :param optimizer: optimizer to use
    :param scheduler: learning rate scheduler
    :param num_classes: number of classes
    :param batch_size: number of samples to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param device: device used by pytorch (cpu ou cuda)
    :return: Updated training loss
    """
    model.train()
    train_metrics = create_metrics_dict(num_classes)

    with tqdm(train_loader, desc=f'Iterating train batches with {device.type}') as _tqdm:
        for batch_index, data in enumerate(_tqdm):
            progress_log.open('a', buffering=1).write(tsv_line(ep_idx, 'trn', batch_index, len(train_loader), time.time()))

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            train_metrics['loss'].update(loss.item(), batch_size)

            if device.type == 'cuda' and debug:
                res, mem = gpu_stats(device=device.index)
                _tqdm.set_postfix(OrderedDict(trn_loss=f'{train_metrics["loss"].val:.2f}',
                                              gpu_perc=f'{res.gpu} %',
                                              gpu_RAM=f'{mem.used / (1024 ** 2):.0f}/{mem.total / (1024 ** 2):.0f} MiB',
                                              lr=optimizer.param_groups[0]['lr'],
                                              img=data['sat_img'].numpy().shape[1:],
                                              smpl=data['map_img'].numpy().shape,
                                              bs=batch_size))

            loss.backward()
            optimizer.step()

    scheduler.step()
    logging.info(f'Training Loss: {train_metrics["loss"].avg:.4f}')
    return train_metrics


def evaluation(eval_loader, model, criterion, num_classes, batch_size, ep_idx, progress_log, batch_metrics=None, dataset='val', device=None, debug=False):
    """
    Evaluate the model and return the updated metrics
    :param eval_loader: data loader
    :param model: model to evaluate
    :param criterion: loss criterion
    :param num_classes: number of classes
    :param batch_size: number of samples to process simultaneously
    :param ep_idx: epoch index (for hypertrainer log)
    :param progress_log: progress log file (for hypertrainer log)
    :param batch_metrics: (int) Metrics computed every (int) batches. If left blank, will not perform metrics.
    :param dataset: (str) 'val or 'tst'
    :param device: device used by pytorch (cpu ou cuda)
    :return: (dict) eval_metrics
    """
    eval_metrics = create_metrics_dict(num_classes)
    model.eval()

    with tqdm(eval_loader, dynamic_ncols=True, desc=f'Iterating {dataset} batches with {device.type}') as _tqdm:
        for batch_index, data in enumerate(_tqdm):
            progress_log.open('a', buffering=1).write(tsv_line(ep_idx, dataset, batch_index, len(eval_loader), time.time()))

            with torch.no_grad():
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels_flatten = labels

                outputs = model(inputs)
                outputs_flatten = outputs

                loss = criterion(outputs, labels)

                eval_metrics['loss'].update(loss.item(), batch_size)

                if (dataset == 'val') and (batch_metrics is not None):
                    # Compute metrics every n batches. Time consuming.
                    assert batch_metrics <= len(_tqdm), f"Batch_metrics ({batch_metrics} is smaller than batch size " \
                        f"{len(_tqdm)}. Metrics in validation loop won't be computed"
                    if (batch_index+1) % batch_metrics == 0:   # +1 to skip val loop at very beginning
                        a, segmentation = torch.max(outputs_flatten, dim=1)
                        eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                             ignore_index=get_key_def("ignore_index", params["training"], None))
                elif dataset == 'tst':
                    a, segmentation = torch.max(outputs_flatten, dim=1)
                    eval_metrics = report_classification(segmentation, labels_flatten, batch_size, eval_metrics,
                                                         ignore_index=get_key_def("ignore_index", params["training"], None))

                _tqdm.set_postfix(OrderedDict(dataset=dataset, loss=f'{eval_metrics["loss"].avg:.4f}'))

                if debug and device.type == 'cuda':
                    res, mem = gpu_stats(device=device.index)
                    _tqdm.set_postfix(OrderedDict(device=device, gpu_perc=f'{res.gpu} %',
                                                  gpu_RAM=f'{mem.used/(1024**2):.0f}/{mem.total/(1024**2):.0f} MiB'))

    logging.info(f"{dataset} Loss: {eval_metrics['loss'].avg}")
    if batch_metrics is not None:
        logging.info(f"{dataset} precision: {eval_metrics['precision'].avg}")
        logging.info(f"{dataset} recall: {eval_metrics['recall'].avg}")
        logging.info(f"{dataset} fscore: {eval_metrics['fscore'].avg}")

    return eval_metrics


if __name__ == '__main__':
    print(f'Start\n')
    parser = argparse.ArgumentParser(description='Training execution')
    parser.add_argument('param_file', metavar='DIR',
                        help='Path to training parameters stored in yaml')
    args = parser.parse_args()
    config_path = Path(args.param_file)
    params = read_parameters(args.param_file)

    main(params, config_path)
    print('End of training')
