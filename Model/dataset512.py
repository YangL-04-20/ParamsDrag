# Copyright 2024 The ParamsDrag Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

import os

import numpy as np
from skimage import io, transform

import torch
from torch.utils.data import Dataset,  DataLoader

class Dataset(Dataset):
  def __init__(self, root, train=True, data_len=0, sparam=3, transform=None):
    self.root = root
    self.train = train
    self.data_len = data_len
    self.sparam = sparam
    self.transform = transform
    if self.train:
      self.params = os.listdir(self.root+'/train')
      self.filenames = [str(self.root) + '/train/' + str(item) for item in self.params]
    else:
      self.params = os.listdir(self.root+'/test')
      self.filenames = [str(self.root) + '/test/' + str(item) for item in self.params]

  # TODO(wenbin): deal with data_len correctly.
  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, index):
    if type(index) == torch.Tensor:
      index = index.item()

    # parameter labels are in filename style
    params = self.params[index][:-4].split('_')
    params = [float(item) for item in params]

    sparams = np.copy(params)
    vparams = np.array([0, 0, 1])
    image = io.imread(self.filenames[index])[:, :, 0:3]

    sample = {"image": image, "sparams": sparams, "vparams": vparams}

    if self.transform:
      sample = self.transform(sample)

    return sample

# data transformation
class Resize(object):
  def __init__(self, size):
    assert isinstance(size, (int, tuple))
    self.size = size

  def __call__(self, sample):
    image = sample["image"]
    sparams = sample["sparams"]
    vparams = sample["vparams"]

    image = transform.resize(
        image, (self.size, self.size), order=1, mode="reflect", preserve_range=True, anti_aliasing=True).astype(np.float32)

    return {"image": image, "sparams": sparams, "vparams": vparams}

class Normalize(object):
  def __call__(self, sample):
    image = sample["image"]
    sparams = sample["sparams"]
    vparams = sample["vparams"]

    # normalize
    image = (image.astype(np.float32) - 127.5) / 127.5
    sparams = sparams.astype(np.float32)
    # parameters are normalized to [-1,1]
    sparams[0] = (sparams[0] - 0.05) / 0.94
    sparams[2] = (sparams[2] - 0.6) / 0.4
    vparams = vparams.astype(np.float32)

    return {"image": image, "sparams": sparams, "vparams": vparams}

class ToTensor(object):
  def __call__(self, sample):
    image = sample["image"]
    sparams = sample["sparams"]
    vparams = sample["vparams"]

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return {"image": torch.from_numpy(image),
            "sparams": torch.from_numpy(sparams),
            "vparams": torch.from_numpy(vparams)}

