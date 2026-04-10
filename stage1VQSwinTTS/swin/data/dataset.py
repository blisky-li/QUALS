import numpy as np
import random
from torch.utils.data import Dataset
import torch

import numpy as np
from torch.utils.data import Dataset
import os

class BLASTDataset(Dataset):

    def __init__(self, mode: str, num_valid_samples: int = None, k_max: int = 3, alpha: float = 1.5, **kwargs):
        super().__init__()
        assert mode in ['train', 'valid', 'test', 'val']
        if mode == 'val':
            mode = 'valid'

        # ------------ 保存初始化参数 ------------
        self.mode = mode
        self.alpha = alpha
        self.num_valid_samples = num_valid_samples
        self.k_max = k_max

        # Keep the dataset root as metadata so DataLoader workers can reopen memmap files.
        self.dataset_root = kwargs.get("data_root", os.environ.get("BLAST_DATA_ROOT", "<DATA_ROOT>/Synth"))
        '''self.shape_path = f"<DATA_ROOT>/datasets/{self.mode}/shape.npy"
        self.data_path = f"<DATA_ROOT>/datasets/{self.mode}/data.dat"'''
        '''self.shape_path = f"/{self.mode}_shape.npy"
        self.data_path = f'/{self.mode}.dat'''


        # ------------ 主进程中首次打开 memmap ------------
        self._open_data()

    # ==================================================
    # 打开 memmap 的函数：主进程 + 每个 worker 都会调用
    # ==================================================
    def _open_data(self):
        shape = np.load(os.path.join(self.dataset_root, self.mode, "shape.npy"))
        
        # 保存元信息
        self.data_path = os.path.join(self.dataset_root, self.mode, "data.dat")
        self.data_shape = tuple(shape)
        self.data_dtype = np.float32

        # 主进程加载 memmap
        self.memmap_data = np.memmap(self.data_path, dtype=self.data_dtype, shape=self.data_shape, mode='r')
        if self.mode == 'valid':
            self.idx_list = random.sample(range(self.memmap_data.shape[0]), self.num_valid_samples)
        print(f"Loaded {self.mode} dataset with shape {self.memmap_data.shape}")

    # ==================================================
    # 让 dataset 可用于多进程 DataLoader 的关键：
    # 移除 memmap，避免序列化失败
    # ==================================================
    def __getstate__(self):
        state = self.__dict__.copy()
        state["memmap_data"] = None     # 不能序列化 memmap
        return state

    # ==================================================
    # 在 worker 内恢复 memmap
    # ==================================================
    def __setstate__(self, state):
        self.__dict__.update(state)
        # 子进程使用元信息重新建立 memmap 映射
        self.memmap_data = np.memmap(
            self.data_path, 
            dtype=self.data_dtype, 
            shape=self.data_shape, 
            mode='r'
        )              # worker 进程重新打开 memmap

    # ==================================================
    # mixup 按你现有代码不动
    # ==================================================
    def mixup(self):
        k = np.random.randint(1, self.k_max + 1)
        sampled_indices = np.random.choice(len(self), size=(k,), replace=True)
        weights = np.random.dirichlet([self.alpha] * k).astype(np.float32)

        time_series_sampled = self.memmap_data[sampled_indices].astype(np.float32)
        nan_mask = np.isnan(time_series_sampled).all(axis=0)

        # normalize
        mean = np.nanmean(time_series_sampled, axis=1, keepdims=True)
        std = np.nanstd(time_series_sampled, axis=1, keepdims=True) + 1e-8
        time_series_sampled = (time_series_sampled - mean) / std
        time_series_sampled = np.nan_to_num(time_series_sampled)

        augmented_batch = np.dot(weights, time_series_sampled)
        augmented_batch[nan_mask] = np.nan

        return augmented_batch

    def __getitem__(self, idx: int) -> tuple:
        # idx is not used in mixup
        if self.mode == 'valid':
            idx = self.idx_list[idx]

        time_series = self.memmap_data[idx]

        std = np.nanstd(time_series)
        # normalize data
        if std < 1e-6:
            std = 1

        time_series = (time_series - np.nanmean(time_series)) / (std + 1e-8)

        return {'time_series': torch.from_numpy(time_series)}

    def __len__(self):
        if self.mode == 'train':
            return self.memmap_data.shape[0]
        else:
            return len(self.idx_list)



class BLASTDatasetold(Dataset):

    def __init__(self, mode: str, num_valid_samples: int = None, k_max: int = 3, alpha : float = 1.5, **kwargs) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'test', 'val']
        if mode == 'val': mode = 'valid'

        self.mode = mode
        self.alpha = alpha
        self.num_valid_samples = num_valid_samples
        self.k_max = k_max

        dataset_root = kwargs.get("data_root", os.environ.get("BLAST_DATA_ROOT", "<DATA_ROOT>/BLAST"))
        shape = np.load(os.path.join(dataset_root, self.mode, "shape.npy"))
        self.memmap_data = np.memmap(
            os.path.join(dataset_root, self.mode, "data.dat"),
            dtype=np.float32,
            shape=tuple(shape),
            mode='r'
        )

        if self.mode == 'valid' and self.num_valid_samples is not None:
            print(f"Using {self.num_valid_samples} samples for {self.mode} dataset")
            import random
            idx_list = random.sample(range(self.memmap_data.shape[0]), self.num_valid_samples)
            self.memmap_data = self.memmap_data[idx_list]

        print(f"Loaded {self.mode} dataset with shape {self.memmap_data.shape}")

    def mixup(self):
        # sampling
        k = np.random.randint(1, self.k_max + 1)
        sampled_indices = np.random.choice(len(self), size=(k), replace=True)
        weights = np.random.dirichlet([self.alpha] * k).astype(np.float32)
        time_series_sampled = self.memmap_data[sampled_indices].copy().astype(np.float32)
        nan_mask = (np.isnan(time_series_sampled)).all(axis=0) # 1: nan, 0: non_nan

        # normalize data
        mean = np.nanmean(time_series_sampled, axis=1, keepdims=True)
        std = np.nanstd(time_series_sampled, axis=1, keepdims=True) + 1e-8
        time_series_sampled = (time_series_sampled - mean) / std
        time_series_sampled = np.nan_to_num(time_series_sampled, nan=0., posinf=0., neginf=0.)
        augmented_batch = np.dot(weights, time_series_sampled)

        # mask data
        augmented_batch[nan_mask] = np.nan

        return augmented_batch

    def __getitem__(self, idx: int) -> tuple:
        # idx is not used in mixup

        # time_series = self.mixup()
        time_series = self.memmap_data[idx].copy().astype(np.float32)

        # normalize data
        time_series = (time_series - np.nanmean(time_series)) / (np.nanstd(time_series) + 1e-8)

        return {'time_series': time_series}

    def __len__(self):
        return self.memmap_data.shape[0]

if __name__ == "__main__":
    dataset = BLASTDatasetMixUp(mode='train')
    a = dataset[0]
    a = 1
