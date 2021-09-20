import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import RandomRotation
from torchvision.transforms.functional import hflip, vflip, rotate

import pandas as pd
from typing import Optional, Dict
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(
            self, fn_img_label: Path, img_dir: Path, n_class: int, transform: Optional[nn.Module],
            v_flip: bool = False, h_flip: bool = False, rotate_degree: int = 0,
            seed=0, binary: bool = False, read_mode: int = ImageReadMode.RGB,
            ):
        '''
        :param fn_img_label: Path. tsv file of image labels with two columns of filenames and corresponding labels.
        :param img_dir: Path. Parent path of image files.
        :param n_class: Int. Number of classes to classify.
        :param transform: Optional[Callable]. Transform function for read images. No transformation if it is None.
        :param seed: Int. Random seed.
        :param binary: Bool. Binarize image if necessary.
        '''
        img_labels = pd.read_csv(fn_img_label).sample(frac=1, random_state=seed).reset_index(drop=True)
        self.filenames = img_labels['filename']
        self.labels = torch.Tensor(img_labels['label'].astype(float))
        self.img_dir = Path(img_dir)
        self.n_class = n_class
        self.transform = transform
        self.binary = binary
        self.read_mode = read_mode
        self.vflip = v_flip
        self.hflip = h_flip
        self.rotate_degree = rotate_degree

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img_path = str(self.img_dir / fn)

        image = read_image(img_path, self.read_mode).float()
        if self.vflip:
            image = vflip(image)
        if self.hflip:
            image = hflip(image)
        if self.rotate_degree:
            image = rotate(image, self.rotate_degree)

        if self.transform:
            image = self.transform(image)
        if self.binary:
            image = image > (image.mean() * 0.45)
            image = image.float()
        floor = (image == 0).float() * 0.01
        image = image / 255 + floor

        label = int(self.labels[idx])
        # Convert labels to lists instead of scalar
        if label >= self.n_class:
            label = self.n_class - 1
        # arr = [0] * self.n_class
        # arr[label] = 1
        return fn, image.float(), label


class SimpleClassifier(nn.Module):
    '''
    A simple classifier consisting of:
    * Cov2D layer * 2
    * Linear layer * 2
    * Softmax layer
    '''
    def __init__(self, input_channels: int, n_class: int, n_cnn: int = 2, n_fc: int = 1, img_size: int = 256):
        super().__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rotater = RandomRotation(30)
        self.random_rotate = False
        self.pool = nn.MaxPool2d(2, 2)
        conv2d_layers = [nn.Conv2d(input_channels, 6, 5, padding=2), ]
        # self.conv1 = nn.Conv2d(input_channels, 6, 5, padding=2)

        for _ in range(n_cnn - 1):
            # cnn = nn.Conv2d(6, 6, 5, padding=2).to(device)
            conv2d_layers.append(nn.Conv2d(6, 6, 5, padding=2))
        self.conv2d_layers = nn.ModuleList(conv2d_layers)
        conv2d_size = int(img_size // (2 ** n_cnn)) ** 2 * 6
        self.fc1 = nn.Linear(conv2d_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_class)
        self.n_cnn = n_cnn
        self.n_fc = n_fc
        self.fc = nn.Linear(conv2d_size, n_class)

    def forward(self, inputs) -> Dict:
        if self.random_rotate:
            inputs = self.rotater(inputs)
        fea = inputs
        # print()
        for cnn in self.conv2d_layers:
            fea = self.pool(cnn(fea).relu())
        # fea = self.pool(self.conv1(inputs).relu())
        # if self.n_cnn > 1:
        #     fea = self.pool(self.conv2(fea).relu())
        fea = torch.flatten(fea, 1)
        if self.n_fc == 1:
            fea = self.fc(fea)
        elif self.n_fc == 3:
            fea = self.fc1(fea).relu()
            fea = self.fc2(fea).relu()
            fea = self.fc3(fea)
        else:
            raise ValueError
        # try:
        #     fea = self.fc1(fea).relu()
        # except Exception as e:
        #     print(fea.shape)
        #     raise e
        # fea = self.fc2(fea).relu()
        # fea = self.fc3(fea).relu()
        labels = fea
        return labels
