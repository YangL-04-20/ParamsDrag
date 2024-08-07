# Copyright 2024 The ParamsDrag Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

import sys
import numpy
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, "./stylegan2")
from torch_utils import misc

# model
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def add_sn(m):
    for name, c in m.named_children():
        m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            return nn.utils.spectral_norm(m, eps=1e-5)
        else:
            return m
    return m

def set_requires_grad(model, requires_grad):
    for submodel in model:
        for param in submodel.parameters():
            param.requires_grad = requires_grad

def set_zerograd(optimizer):
    for sub_optimizer in optimizer:
        sub_optimizer.zero_grad(set_to_none=True)

def set_step(optimizer):
    for sub_optimizer in optimizer:
        sub_optimizer.step()

def remove_inf(model):
    for submodel in model:
        for param in submodel.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

def getStat(train_data):
    print('Compute mean and variance for training data.')
    mean = numpy.array(mean)
    std = numpy.array(std)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=False, num_workers=40, pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X in train_loader:
        X = X["image"]
        for c in range(3):
            mean[c] += X[:, c, :, :].mean()
            std[c] += X[:, c, :, :].std()
    mean.div_(len(train_loader))
    std.div_(len(train_loader))
    return mean.numpy(), std.numpy()

# Red sea -- mean:  [0.9230321  0.90690917 0.875938  ] std:  [0.33231997 0.34463495 0.43401787]
# Thick Universe -- mean:  [-0.8271422  -0.6777301  -0.46108523] std:  [0.19843157 0.2517174  0.41449106]



# VGG19 network for content-loss computation
class VGG19(nn.Module):
  def __init__(self, layer="relu1_2"):
    super(VGG19, self).__init__()
    features = models.vgg19(pretrained=True).features

    self.layer_dict = {"relu1_1": 2, "relu1_2": 4,
                       "relu2_1": 7, "relu2_2": 9,
                       "relu3_1": 12, "relu3_2": 14}

    self.layer = layer
    self.subnet = nn.Sequential()
    for i in range(self.layer_dict[self.layer]):
      self.subnet.add_module(str(i), features[i])

    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    out = self.subnet(x)
    return out


# canny-alg for edge-loss computation
def canny_edge_detection(image, sigmaX, low_threshold, high_threshold):
    gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(sigmaX, sigmaX))
    image_blurred = gaussian_blur(image)

    sobel_x = (torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float()).unsqueeze(0).unsqueeze(0).to("cuda")
    sobel_y = (torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float()).unsqueeze(0).unsqueeze(0).to("cuda")
    gradient_x = F.conv2d(image_blurred, sobel_x, padding=1)
    gradient_y = F.conv2d(image_blurred, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = torch.atan2(gradient_y, gradient_x)
    
    gradient_magnitude_suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    edge_map = double_threshold(gradient_magnitude_suppressed, low_threshold, high_threshold)
    
    edge_map_final = edge_tracking(edge_map)
    
    return edge_map_final

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    height, width = gradient_magnitude.shape[2:]

    nms_result = torch.zeros_like(gradient_magnitude)

    gradient_direction = gradient_direction * 180.0 / torch.tensor(3.14159265358979323846)

    result = torch.zeros_like(gradient_magnitude)
    for index in range(gradient_magnitude.shape[0]):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                mag = gradient_magnitude[index, 0, y, x].item()
                angle = gradient_direction[index, 0, y, x].item()

                angle = angle % 180
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q = gradient_magnitude[index, 0, y, x + 1].item()
                    r = gradient_magnitude[index, 0, y, x - 1].item()
                elif (22.5 <= angle < 67.5):
                    q = gradient_magnitude[index, 0, y - 1, x + 1].item()
                    r = gradient_magnitude[index, 0, y + 1, x - 1].item()
                elif (67.5 <= angle < 112.5):
                    q = gradient_magnitude[index, 0, y + 1, x].item()
                    r = gradient_magnitude[index, 0, y - 1, x].item()
                else:
                    q = gradient_magnitude[index, 0, y - 1, x - 1].item()
                    r = gradient_magnitude[index, 0, y + 1, x + 1].item()

                if mag >= q and mag >= r:
                    nms_result[index, 0, y, x] = mag
                else:
                    nms_result[index, 0, y, x] = 0
    return nms_result

def double_threshold(gradient_magnitude, low_threshold, high_threshold):
    strong_edges = (gradient_magnitude >= high_threshold).float()
    weak_edges = ((gradient_magnitude < high_threshold) & (gradient_magnitude >= low_threshold)).float()

    return strong_edges, weak_edges

def edge_tracking(edge_map):
    strong_edges, weak_edges = edge_map

    edge_map_final = torch.zeros_like(strong_edges)

    edge_map_final += strong_edges

    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    edge_map = torch.zeros_like(strong_edges)
    for index in range(weak_edges.shape[0]):
        for y in range(1, weak_edges.shape[2] - 1):
            for x in range(1, weak_edges.shape[3] - 1):
                if weak_edges[index, 0, y, x] == 1:
                    for dy, dx in offsets:
                        ny, nx = y + dy, x + dx
                        if strong_edges[index, 0, ny, nx] == 1:
                            edge_map_final[index, 0, y, x] = 1
                            break
    return edge_map_final