import numpy as np
import torch
import torch.utils.data as data
import os
import cv2
import struct


def read_bin(filename, shape):
    with open(filename, 'rb') as f:
        data = struct.unpack('%df' % (2 * (shape[0]) * (shape[1])), f.read())
        return np.asarray(data).reshape(2, shape[0], shape[1])


class SpeckleDataset(data.Dataset):

    def __init__(self, root_dir, total):
        super(SpeckleDataset, self).__init__()

        self.root_dir = root_dir
        self.size = total

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ref = cv2.imread(os.path.join(self.root_dir, 'REF%05d.bmp' % idx), cv2.IMREAD_GRAYSCALE)
        tar = cv2.imread(os.path.join(self.root_dir, 'TAR%05d.bmp' % idx), cv2.IMREAD_GRAYSCALE)

        h, w = ref.shape
        deform = read_bin(os.path.join(self.root_dir, 'DEF%05d.bin' % idx), (h, w))
        ref = ref[np.newaxis, ...].astype(np.float32)
        tar = tar[np.newaxis, ...].astype(np.float32)
        return torch.Tensor(ref), torch.Tensor(tar), torch.Tensor(deform), torch.ones((h, w))


def build_train_dataset(args):
    """ Create the data loader for the corresponding training set """
    if args.stage == 'speckle':
        train_dataset = SpeckleDataset(r'D:\train', 82)

    else:
        raise ValueError(f'stage {args.stage} is not supported')

    return train_dataset
