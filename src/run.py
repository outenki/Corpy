#!/usr/bin/env python
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset, random_split
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode

import torch
from torch import nn
from lib import ImageDataset, SimpleClassifier
from utils import run_classifier, set_logger, draw_curves

import argparse
import logging
from pandas.core.frame import DataFrame
# import git

from pathlib import Path

from nlptools import utils as U
import numpy as np


def save_results(results, dn):
    img_fn = results['filenames']
    truth = results['truth']
    pred_label = results['pred_label']
    pred_dist = results['pred_dist']
    acc = results['accuracy']
    f1 = results['f1']
    p = results['precision']
    r = results['recall']

    res = DataFrame(
        {
            "img_fn": img_fn,
            "true_label": truth,
            "pred_label": pred_label,
        }
    )
    fn = output_path / f'{dn}_results.tsv'
    logger.info(f"Save {dn} data to: {fn}")
    res.to_csv(fn, index=False, columns={'img_fn', 'true_label', 'pred_label'}, sep='\t')

    pred_dist = DataFrame(np.column_stack([pred_dist, truth, pred_label]), img_fn, list(range(n_class))+['truth', 'pred'])
    fn = output_path / f'{dn}_dist.tsv'
    logger.info(f"Save {dn} data to: {fn}")
    pred_dist.to_csv(fn, sep='\t', float_format="%.3f")

    fn = output_path / f'{dn}_eval.tsv'
    with open(fn, 'w') as f:
        f.write(f"{acc:.3f}\t{f1:.3f}\t{p:.3f}\t{r:.3f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', '-d', dest='data_path', type=str, default='', required=True, help='The path to train/test data files.'
    )
    parser.add_argument('--read-mode', '-rm', dest='read_mode', type=str, choices={'rgb', 'rgba', 'gray'})
    parser.add_argument('--binary-image', '-bi', dest='binary_image', action='store_true', help='Binarize image')
    parser.add_argument('--val-data', dest='val_data', type=float, help='Choose best proportion of val data (from 0 to 1).')
    parser.add_argument('--reshape', '-rs', dest='reshape', action='store_true', help='Reshape image to 256x256')
    parser.add_argument('--ext-data', dest='ext_data', action='store_true', help='Extend data by rotating and fliping')
    parser.add_argument('--random-rotate', dest='random_rotate', action='store_true', help='Random rotate in training')
    parser.add_argument('--learning-rate', '-lr', dest='lr', type=float, metavar='<float>', help='Learning rate')
    parser.add_argument('--batch-size', dest='batch_size', type=int, metavar='<int>', help='Batch size')
    parser.add_argument('--ok-weight', dest='ok_weight', type=float, metavar='<float>', help='Weight of loss for OK class')
    parser.add_argument('--seed', dest='seed', type=int, metavar='<int>', help='Random seed')
    parser.add_argument('--num-cnn', dest='n_cnn', type=int, metavar='<int>', help='Number of CNN layers')
    parser.add_argument('--num-fc', dest='n_fc', type=int, metavar='<int>', choices={1, 3}, help='Number of FC layer')
    parser.add_argument('--epoch', dest='epoch', type=int, metavar='<int>', help='Number epochs')
    parser.add_argument('--class-num', dest='n_class', type=int, metavar='<int>', help='Number of classes')
    parser.add_argument('--output-path', '-o', dest='output_path', type=str, help='Path to save results')
    return parser.parse_args()


args = parse_arguments()
random_seed = args.seed
torch.random.manual_seed(random_seed)

output_path = Path(args.output_path)
output_path.mkdir(parents=True, exist_ok=True)

# logging
logger = logging.getLogger(__name__)
set_logger(output_path)
U.print_args(args)

# prepare data
batch_size = args.batch_size
n_class = args.n_class
data_path = Path(args.data_path)
train_path = data_path / 'train'
train_label = data_path / 'train.csv'
test_path = data_path / 'test'
test_label = data_path / 'test.csv'

# Settings to read images
if args.read_mode == 'rgb':
    read_mode = ImageReadMode.RGB
    input_channels = 3
elif args.read_mode == 'rgba':
    read_mode = ImageReadMode.RGB_ALPHA
    input_channels = 4
elif args.read_mode == 'gray':
    read_mode = ImageReadMode.GRAY
    input_channels = 1
else:
    read_mode = None
    raise ValueError

if args.reshape:
    transform = transforms.Resize((256, 256))
    img_size = 256
else:
    img_size = 1024
    transform = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"device: {device}")

# Load data
train_ds = [ImageDataset(
    fn_img_label=train_label, img_dir=train_path, n_class=n_class,
    transform=transform, read_mode=read_mode, binary=args.binary_image)]

# Extend dataset
if args.ext_data:
    # v_flip
    train_ds.append(ImageDataset(
        fn_img_label=train_label, img_dir=train_path, n_class=n_class, transform=transform,
        v_flip=True, read_mode=read_mode, binary=args.binary_image))
    # h_flip
    train_ds.append(ImageDataset(
        fn_img_label=train_label, img_dir=train_path, n_class=n_class, transform=transform,
        h_flip=True, read_mode=read_mode, binary=args.binary_image))
    # rotate
    for degree in (90, 180, 270):
        train_ds.append(ImageDataset(
            fn_img_label=train_label, img_dir=train_path, n_class=n_class, transform=transform,
            rotate_degree=degree, read_mode=read_mode, binary=args.binary_image))
train_ds = ConcatDataset(train_ds)

# Sample from training dta for validation
if 0 < args.val_data < 1:
    data_size = len(train_ds)
    val_size = int(data_size * args.val_data)
    train_size = data_size - val_size
    train_ds, val_ds = random_split(train_ds, (train_size, val_size))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
else:
    # No valization data then take the parameter at the last epoch
    val_dl = list()
    val_ds = list()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = ImageDataset(
    fn_img_label=test_label, img_dir=test_path, n_class=n_class,
    transform=transform, read_mode=read_mode, binary=args.binary_image
)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
logger.info(f"Size of train/val/test data: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

classifier = SimpleClassifier(
    input_channels=input_channels, n_class=n_class, img_size=img_size, n_cnn=args.n_cnn, n_fc=args.n_fc
).to(device)

# Initialize criterion with weights
class_weights = [1.0] * n_class
class_weights[0] = args.ok_weight
class_weights = torch.Tensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-3)

logger.info("Training ...")
epochs = args.epoch

loss_log = list()
f1_log = list()
f_best_weights = output_path / 'best_weights'
logger.info("Saving initial weights to %s", f_best_weights)
torch.save(classifier.state_dict(), f_best_weights)

best_dev_loss = float('inf')
best_test_loss = float('inf')
best_test_accuracy = float('inf')
for epoch in range(1, epochs + 1):
    logger.info(f'======= epoch {epoch}/{epochs}')
    logger.info('train:')
    logger.info(f"Set random_rotate to {args.random_rotate}")
    classifier.random_rotate = args.random_rotate
    train_results = run_classifier(
        model=classifier, data=train_dl, device=device, criterion=criterion, optimizer=optimizer, train=True
    )
    with torch.no_grad():
        classifier.random_rotate = False
        logger.info("Set random_rotate to False")

        if len(val_ds) > 0:
            logger.info('val:')
            val_results = run_classifier(
                model=classifier, data=val_dl, device=device, criterion=criterion, optimizer=optimizer, train=False
            )
        else:
            val_results = {'loss': -1}
        logger.info('test:')
        test_results = run_classifier(
            model=classifier, data=test_dl, device=device, criterion=criterion, optimizer=optimizer, train=False
        )
    loss_log.append([epoch, train_results['loss'], val_results['loss'], test_results['loss']])
    f1_log.append([epoch, train_results['f1'], val_results['f1'], test_results['f1']])
    logger.info(
        f"train_loss: {train_results['loss']:.3f} - acc: {train_results['accuracy']:.3f}"
        f" - f1: {train_results['f1']:.3f} - p: {train_results['precision']:.3f} - r: {train_results['recall']:.3f}"
    )
    if val_ds:
        logger.info(
            f"val_loss: {val_results['loss']:.3f} - acc: {val_results['accuracy']:.3f}"
            f" - f1: {val_results['f1']:.3f} - p: {val_results['precision']:.3f} - r: {val_results['recall']:.3f}"
        )
    logger.info(
        f"test_loss: {test_results['loss']:.3f} - acc: {test_results['accuracy']:.3f}"
        f" - f1: {test_results['f1']:.3f} - p: {test_results['precision']:.3f} - r: {test_results['recall']:.3f}"
    )
    if val_ds and best_dev_loss > val_results['loss']:
        best_dev_loss = val_results['loss']
        logger.info("Update best weights")
        torch.save(classifier.state_dict(), f_best_weights)
    print()
if not val_ds:
    logger.info("Update best weights in the end.")
    torch.save(classifier.state_dict(), f_best_weights)

f_last_weights = output_path / 'last_weights'
logger.info("Save last weights.")
torch.save(classifier.state_dict(), f_last_weights)

# save loss log
loss_log = np.array(loss_log).astype(float)
np.savetxt(output_path / 'loss.log', loss_log, delimiter='\t', fmt='%.4f')
# visualization of loss
f = output_path / 'loss.png'
logger.info("Save loss_log to %s", f)
draw_curves(f, loss_log.T[1:], ['train', 'dev', 'test'], 'Loss of train/dev/test data', 'epochs', 'Loss')

# save f1 log
f1_log = np.array(f1_log).astype(float)
np.savetxt(output_path / 'f1.log', f1_log, delimiter='\t', fmt='%.4f')
# visualization of f1
f = output_path / 'f1.png'
logger.info("Save loss_log to %s", f)
draw_curves(f, f1_log.T[1:], ['train', 'dev', 'test'], 'Loss of train/dev/test data', 'epochs', 'Loss')

with torch.no_grad():
    logger.info(f"Loading best_weights from {f_best_weights} ...")
    classifier.load_state_dict(torch.load(f_best_weights))
    logger.info('train:')
    train_results = run_classifier(
        model=classifier, data=train_dl, device=device, criterion=criterion, optimizer=optimizer, train=False
    )
    save_results(train_results, "train")
    logger.info(
        f"train_loss: {train_results['loss']:.3f} - train_acc: {train_results['accuracy']:.3f}"
        f" - f1: {train_results['f1']:.3f} - p: {train_results['precision']:.3f} - r: {train_results['recall']:.3f}"
    )

    if len(val_ds) > 0:
        logger.info('val:')
        val_results = run_classifier(
            model=classifier, data=val_dl, device=device, criterion=criterion, optimizer=optimizer, train=False
        )
        save_results(val_results, "val")
        logger.info(
            f"valn_loss: {val_results['loss']:.3f} - val_acc: {val_results['accuracy']:.3f}"
            f" - f1: {val_results['f1']:.3f} - p: {val_results['precision']:.3f} - r: {val_results['recall']:.3f}"
        )

    logger.info('test:')
    test_results = run_classifier(
        model=classifier, data=test_dl, device=device, criterion=criterion,
        optimizer=optimizer, train=False
    )
    save_results(test_results, "test")
    logger.info(
        f"test_loss: {test_results['loss']:.3f} - test_acc: {test_results['accuracy']:.3f}"
        f" - f1: {test_results['f1']:.3f} - p: {test_results['precision']:.3f} - r: {test_results['recall']:.3f}"
    )

fn = output_path / 'kernel.png'
logger.info(f"Saving kernel of the first Conv2D layer to {fn}...")
kernel = classifier.conv2d_layers[0].weight.data.cpu()
kernel = kernel - kernel.min()
kernel = kernel / kernel.max()
filter_img = torchvision.utils.make_grid(kernel, n_row=1)
plt.imshow(filter_img.permute(1, 2, 0))
plt.savefig(fn)
