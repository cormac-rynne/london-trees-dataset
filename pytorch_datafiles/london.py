import numpy as np
import os
from torch.utils import data
from scipy import io as sio
from PIL import Image
import h5py
from .utils import read_json_filelist
import pandas as pd


class London(data.Dataset):
    def __init__(self, json_filepath, mode='', main_transform=None, img_transform=None, gt_transform=None):

        self.file_lst = read_json_filelist(json_filepath)
        if mode.lower() == 'train':
            self.file_lst *= 2
        self.num_samples = len(self.file_lst)

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        filepaths = self.file_lst[index]
        img, den = self.read_image_and_gt(filepaths)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, filepaths):
        img_filepath = filepaths[0]
        den_filepath = filepaths[1]
        img = Image.open(img_filepath)

        if img.mode == 'L':
            img = img.convert('RGB')

        _, ext = os.path.splitext(den_filepath)

        # Read file
        if ext == '.mat':
            den = sio.loadmat(den_filepath)
            den = den['map']
        elif ext == '.csv':
            den = pd.read_csv(den_filepath, header=None).values
        elif ext == '.h5':
            with h5py.File(den_filepath, "r") as f:
                den = f['density'][:]

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples
