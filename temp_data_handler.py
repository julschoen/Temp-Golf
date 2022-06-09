import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os


class DATA(Dataset):
  def __init__(self, path): 
    self.files = np.load(path)['x']
    self.len = self.files.shape[0]

  def __getitem__(self, index):
      x = self.files[index]
      ind = np.sort(np.random.choice(x.shape[0], 3, replace=False))
      xs = x[ind]
      xs = np.clip(xs, -1, 1)
      return torch.from_numpy(xs).float().squeeze(), torch.Tensor(ind)

  def __len__(self):
      return self.len
