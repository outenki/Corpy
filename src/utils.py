from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_fscore_support as sk_recall_precision_f1
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, List

import os
import re
import logging
import tqdm


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'
    BHEADER = BOLD + '\033[95m'
    BOKBLUE = BOLD + '\033[94m'
    BOKGREEN = BOLD + '\033[92m'
    BWARNING = BOLD + '\033[93m'
    BFAIL = BOLD + '\033[91m'
    BUNDERLINE = BOLD + '\033[4m'
    BWHITE = BOLD + '\033[37m'
    BYELLOW = BOLD + '\033[33m'
    BGREEN = BOLD + '\033[32m'
    BBLUE = BOLD + '\033[34m'
    BCYAN = BOLD + '\033[36m'
    BRED = BOLD + '\033[31m'
    BMAGENTA = BOLD + '\033[35m'
    BBLACK = BOLD + '\033[30m'

    @staticmethod
    def cleared(s):
        return re.sub(r"\033\[[0-9][0-9]?m", "", s)


def red(message):
    return BColors.RED + str(message) + BColors.ENDC


def b_red(message):
    return BColors.BRED + str(message) + BColors.ENDC


def blue(message):
    return BColors.BLUE + str(message) + BColors.ENDC


def b_blue(message):
    return BColors.BBLUE + str(message) + BColors.ENDC


def b_yellow(message):
    return BColors.BYELLOW + str(message) + BColors.ENDC


def green(message):
    return BColors.GREEN + str(message) + BColors.ENDC


def b_green(message):
    return BColors.BGREEN + str(message) + BColors.ENDC


def set_logger(out_dir=None, fn='log.txt'):
    console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + '%(asctime)s  (%(name)s) %(message)s'
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)
    if out_dir:
        file_format = '[%(levelname)s] %(asctime)s (%(name)s) %(message)s'
        log_file = logging.FileHandler(os.path.join(out_dir, fn), mode='w')
        log_file.setLevel(logging.INFO)
        log_file.setFormatter(logging.Formatter(file_format))
        logger.addHandler(log_file)


def run_classifier(model: nn.Module, data: DataLoader, train: bool, device, optimizer, criterion):
    if train:
        model.train()
    else:
        model.eval()

    data_size = 0
    epoch_loss = 0
    pred_label = list()
    pred_dist = list()
    truth = list()
    filenames = list()

    # limit = 0
    for batch in tqdm.tqdm(data, ncols=100):
        # limit += 1
        # if limit > 2:
        #     break
        optimizer.zero_grad()

        batch_size = len(batch[0])
        data_size += batch_size

        fn_img, inputs, y = batch
        # inputs, y = batch
        inputs = inputs.to(device)
        y = y.to(device)

        # The output is distribution of prob
        dist = model(inputs)
        _, label = torch.max(dist, 1)

        truth.append(y)
        pred_dist.append(dist)
        pred_label.append(label)
        filenames += fn_img
        # filenames += y.tolist()

        # bad_mask = (y > 0).float()
        # loss_bad = criterion(dist * bad_mask.unsqueeze(1), y * bad_mask)
        loss = criterion(dist, y)
        # print(f"loss: {loss:.3f}")
        if train:
            loss.backward()
            optimizer.step()
            # scheduler.step()
        epoch_loss += batch_size * loss.item()

    global_loss = epoch_loss / data_size

    truth = torch.cat(truth).cpu().numpy()
    pred_label = torch.cat(pred_label).detach().cpu().numpy()
    pred_dist = torch.cat(pred_dist).softmax(1).detach().cpu().numpy()
    p, r, f1, _ = sk_recall_precision_f1(truth, pred_label, average='binary')
    # print(f1, p, r)
    correct = (pred_label == truth).sum().item()
    accuracy = correct / data_size

    bad_correct = (truth > 0)[pred_label == truth].sum().item()
    bad = (truth > 0).sum().item()
    bad_accuracy = bad_correct / bad

    good_correct = (truth == 0)[pred_label == truth].sum().item()
    good = (truth == 0).sum().item()
    good_accuracy = good_correct / good

    return {
        'loss': global_loss, 'pred_dist': pred_dist, 'pred_label': pred_label,
        'filenames': filenames, 'truth': truth, 'accuracy': accuracy,
        'bad_accuracy': bad_accuracy, 'good_accuracy': good_accuracy,
        'f1': f1, 'precision': p, 'recall': r
        }


def draw_curves(
        output_loc: str,
        data_list: Iterable[np.ndarray],
        legend_list: List[str],
        title: str, xlabel: str, ylabel: str):
    ''' Draw curves
    :param output_loc: str, path to the file to output
    :param legend_list: List[str], names of each curve
    :param data_list: List[np.ndarray], datas to draw. All of them
        are expected to have same length.
    :param title: str, title of the figure
    :param xlabel: str, label of x axel
    :param ylabel: str, label of y axel
    '''
    for data in data_list:
        plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend_list, loc='upper left')
    plt.savefig(output_loc)
    plt.clf()
