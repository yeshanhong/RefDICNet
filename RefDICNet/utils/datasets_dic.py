import numpy as np
import torch
import torch.utils.data as data
import os
import cv2
import struct
import scipy

def read_bin(filename, shape):
    with open(filename, 'rb') as f:
        data = struct.unpack('%df' % (2 * (shape[0]) * (shape[1])), f.read())
        return np.asarray(data).reshape(2, shape[0], shape[1])

# def read_mat(filename, shape):
#     with open(filename, 'rb') as f:
#         data = struct.unpack('%df' % (2 * (shape[0]) * (shape[1])), f.read())
#         return np.asarray(data).reshape(2, shape[0], shape[1])
def read_mat(filename, shape):
    mat_data = scipy.io.loadmat(filename)  # 加载 .mat 文件
    return mat_data.get('uu')  # 替换 'data' 为你需要的变量名

def read_npy(filename, shape):
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
        ref = cv2.imread(os.path.join(self.root_dir, 'ref', 'REF%05d.bmp' % idx), cv2.IMREAD_GRAYSCALE)
        tar = cv2.imread(os.path.join(self.root_dir, 'def', 'TAR%05d.bmp' % idx), cv2.IMREAD_GRAYSCALE)
        h, w = ref.shape
        deform = read_bin(os.path.join(self.root_dir, 'dis', 'DEF%05d.bin' % idx), (h, w))
        deform = deform.transpose(1, 2, 0)
        deform = torch.from_numpy(deform).permute(2, 0, 1).float()
        # mat_file = scipy.io.loadmat(os.path.join(self.root_dir, 'dis', 'displacement%04d.mat' % idx))
        # 假设.mat文件中位移场存储在名为'deform'的变量中
        # deform = mat_file['uu']
        ref = ref[np.newaxis, ...].astype(np.float32)
        tar = tar[np.newaxis, ...].astype(np.float32)
        return torch.Tensor(ref), torch.Tensor(tar), torch.Tensor(deform), torch.ones((h, w))


class WYDataset(data.Dataset):

    def __init__(self, root_dir, total):
        super(WYDataset, self).__init__()

        self.root_dir = root_dir
        self.size = total

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ref = cv2.imread(os.path.join(self.root_dir, 'ref', 'reference%04d.bmp' % idx), cv2.IMREAD_GRAYSCALE)
        tar = cv2.imread(os.path.join(self.root_dir, 'def', 'deformation%04d.bmp' % idx), cv2.IMREAD_GRAYSCALE)
        h, w = ref.shape
        deform = read_mat(os.path.join(self.root_dir, 'dis', 'displacement%04d.mat' % idx), (h, w))
        # mat_file = scipy.io.loadmat(os.path.join(self.root_dir, 'dis', 'displacement%04d.mat' % idx))
        # 假设.mat文件中位移场存储在名为'deform'的变量中
        # deform = mat_file['uu']
        ref = ref[np.newaxis, ...].astype(np.float32)
        tar = tar[np.newaxis, ...].astype(np.float32)
        return torch.Tensor(ref), torch.Tensor(tar), torch.Tensor(deform), torch.ones((h, w))

def build_train_dataset(args):
    """ Create the data loader for the corresponding training set """
    if args.stage == 'speckle':
        train_dataset = SpeckleDataset(r'E:\YSH\Data\Data_DICtr\train', 4000)
    elif args.stage == 'wangyin':
        train_dataset = WYDataset(r'E:\YSH\Data\Data_DICtr\Data_wangyin', 10000)
    else:
        raise ValueError(f'stage {args.stage} is not supported')

    return train_dataset
