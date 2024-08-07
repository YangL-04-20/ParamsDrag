# Copyright 2024 The ParamsDrag Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "../Model/stylegan2")
sys.path.insert(0, "../Model")

from network import Generator
import legacy
import dnnlib
from torch_utils import misc
import utils

import math
import cv2
import numpy as np


# cal function

def dequeue(a, center, threshhold):
    # set threshhold in color range
    if math.sqrt((a[0][0] - center[0])**2 + (a[0][1] - center[1])**2) <= threshhold:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if [a[0][0] + i, a[0][1] + j] not in a and 0 <= a[0][0] + i <= 511 and 0 <= a[0][1] + j <= 511:
                    a.append([a[0][0] + i, a[0][1] + j])
    a.pop(0)
    return a

def check_within_range(old_array, new_array, bound):
    if old_array > new_array:
        diff = old_array - new_array
    else:
        diff = new_array - old_array
        
    within_range = diff <= bound
    return within_range.all()

def search(hsv_img, coord, bound, threshhold):
    patch = []
    queue = []
    queue.append(coord)
    center = hsv_img[queue[0][0]][queue[0][1]]

    while(len(queue) != 0):
        pixel_now = hsv_img[queue[0][0]][queue[0][1]]
        # only use hsv-image's first dimension
        if check_within_range(center[0], pixel_now[0], bound) and queue[0] not in patch:
            patch.append(queue[0])
            dequeue(queue, coord, threshhold)
        else:
            queue.pop(0)
    return patch

def search_patch(img, center, bound, threshhold):
    if not hasattr(search_patch, "counter"):
        search_patch.counter = 0
    search_patch.counter += 1
    
    img = (np.array((img.squeeze(0).permute(1,2,0).detach().cpu() * 127.5 + 128).clamp(0, 255))).astype('uint8')
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    center = [int(number) for number in center]
    
    p = search(hsv_img, center, bound, threshhold)

    return(p)

def structure_supervision(patch, handle_points_init, handle_points_new, img0, img2, resolution, device):
    # supervise pixels in the patch, too little means structure doesn't exist
    if len(patch) < 10:
        print("structure dismissed")
        return True
    # supervise hsv-image's color, out of range means structure has changed
    for i in len(handle_points_init):
        init_coordinates = torch.nonzero(utils.create_square_mask(resolution.shape[2], resolution.shape[3], center=handle_points_init[i].tolist(), radius=2).to(device))
        new_coordinates = torch.nonzero(utils.create_square_mask(resolution.shape[2], resolution.shape[3], center=handle_points_new[i].tolist(), radius=2).to(device))
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        init_h_value = torch.stack([img0[coord[0], coord[1]] for coord in init_coordinates])
        new_h_value = torch.stack([img2[coord[0], coord[1]] for coord in new_coordinates])
        if (new_h_value - init_h_value).mean() > 15:
            print("structure changed")
            return True


# params optimizer drag gan

def load_pretrained_weights(model, checkpoint):
    import collections
    state_dict = checkpoint['G_model']
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model

def load_model(  
    network_pkl: str = resume,
    device: torch.device = torch.device("cuda"),
    fp16: bool = True,
) -> torch.nn.Module:
    print('Loading networks from "%s"...' % network_pkl)
    G = Generator(512, 512, 512, 3, 3)
    checkpoint = torch.load(network_pkl)
    G = load_pretrained_weights(G, checkpoint)
    G.to(device).eval()
    print("===>G load done")
    for param in G.parameters():
        param.requires_grad_(False)
    return G

def register_hook(G):
    # Create a new attribute called "activations" for the Generator class
    # This will be a list of activations from each layer
    G.__setattr__("activations", None)

    # Forward hook to collect features
    def hook(module, input, output):
        G.activations = output

    # Apply the hook to the 7th layer (256x256)
    for i, (name, module) in enumerate(G.synthesis.named_children()):
        if i == 7:
            module.register_forward_hook(hook)
    return G

def forward_G(
        G: torch.nn.Module,
        sparams,
        vparams,
):
    register_hook(G)
    img = G(sparams, vparams, force_fp32=True)
    return img, G.activations[0]

def drag_gan(
        sparams,
        vparams,
        G,
        handle_points,
        target_points,
        mask,
        max_iters,
        Rm,
        bound,
        threshhold,
        d,
        lr
):
    device = torch.device("cuda")

    handle_points0 = copy.deepcopy(handle_points)
    handle_points = torch.stack(handle_points).to(device)
    handle_points0 = torch.stack(handle_points0).to(device)
    target_points = torch.stack(target_points).to(device)

    img0, F0 = forward_G(G, sparams, vparams)

    target_resolution = img0.shape[-1]
    F0_resized = torch.nn.functional.interpolate(
        F0,
        size=(target_resolution, target_resolution),
        mode="bilinear",
        align_corners=True,
    ).detach()

    # In this case, we only optimize sparams
    vparams.requires_grad_(False)
    sparams_to_optimize = sparams.clone().to(device)
    sparams_to_optimize.requires_grad_(True)
    
    init_param = "lr:" + str(lr)+" threshhold:"+str(threshhold)+" bound:"+str(bound)+" Rm:"+str(Rm)+" d:"+str(d)
    print(init_param)
    print("=====> Start Loop")

    optimizer = torch.optim.Adam([sparams_to_optimize], lr=lr)

    for iter in range(max_iters):
        start = time.perf_counter()

        # motion supervision
        img1, F1 = forward_G(G, sparams_to_optimize, vparams)
        F_resized = torch.nn.functional.interpolate(
            F1,
            size=(target_resolution, target_resolution),
            mode="bilinear",
            align_corners=True,
        )
        optimizer.zero_grad()
        loss, patch, plus_patch = feature_supervison(handle_points, target_points, img1, F_resized, bound, threshhold, device)
        loss.backward()
        optimizer.step()
        
        # Point Track
        img2, F2 = forward_G(G, sparams_to_optimize, vparams)
        F_resized = torch.nn.functional.interpolate(
            F2,
            size=(target_resolution, target_resolution),
            mode="bilinear",
            align_corners=True,
        )
        with torch.no_grad():
            track_handle_points, _ = feature_tracking(F_resized, F0_resized, handle_points, handle_points0, Rm, device)
            # control sparams in range
            if torch.any((sparams_to_optimize < 0) | (sparams_to_optimize > 1)):
                sparams_to_optimize[0] = torch.clamp(sparams_to_optimize[0], 0 , 1)
        
        print(iter+1, "loss:", "{:.5}".format(loss.item()), "sp:({:.5}, {:.5}, {:.5})".format(sparams_to_optimize.tolist()[0][0], sparams_to_optimize.tolist()[0][1],
                             sparams_to_optimize.tolist()[0][2]), "point:",handle_points0.tolist()[0], track_handle_points.tolist()[0], target_points.tolist()[0])
        img1 = utils.tensor_to_PIL(img1)
        if torch.allclose(handle_points, target_points, atol=d):
            print("Iteration termination")
            break
        if structure_supervision(patch, handle_points0, track_handle_points, img0, img2, target_resolution, device):
            break
        handle_points = track_handle_points
        yield img1, sparams_to_optimize, track_handle_points

def feature_supervison(handle_points, target_points,img, F, bound, threshhold, device):
    loss = 0
    n = len(handle_points)
    for i in range(n):
        target2handle = target_points[i] - handle_points[i]
        d_i = target2handle / (torch.norm(target2handle) + 1e-7)
        if torch.norm(d_i) > torch.norm(target2handle):
            d_i = target2handle

        # mask = utils.create_circular_mask(F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r1).to(device)
        # coordinates = torch.nonzero(mask).float()
        
        # using patch search, rather than circle-mask, to find accurate structure
        coordinates = torch.tensor(search_patch(img, handle_points[i].tolist(), bound, threshhold)).float().to(device)
        mask = torch.zeros(F.shape[2], F.shape[3]).bool().to(device)
        for x, y in coordinates:
            x = int(x)
            y = int(y)
            mask[y, x] = 1

        # Shift the coordinates in the direction d_i
        shifted_coordinates = coordinates + d_i[None]

        h, w = F.shape[2], F.shape[3]

        # Extract features in the mask region and compute the loss
        F_qi = F[:, :, mask]  # shape: [C, H*W]

        # Sample shifted patch from F
        normalized_shifted_coordinates = shifted_coordinates.clone()
        normalized_shifted_coordinates[:, 0] = (2.0 * shifted_coordinates[:, 0] / (h - 1)) - 1  # for height
        normalized_shifted_coordinates[:, 1] = (2.0 * shifted_coordinates[:, 1] / (w - 1)) - 1  # for width
        # Add extra dimensions for batch and channels (required by grid_sample)
        normalized_shifted_coordinates = normalized_shifted_coordinates.unsqueeze(0).unsqueeze(0)  # shape [1, 1, num_points, 2]
        normalized_shifted_coordinates = normalized_shifted_coordinates.flip(-1)  # grid_sample expects [x, y] instead of [y, x]
        normalized_shifted_coordinates = normalized_shifted_coordinates.clamp(-1, 1)

        # Use grid_sample to interpolate the feature map F at the shifted patch coordinates
        F_qi_plus_di = torch.nn.functional.grid_sample(F, normalized_shifted_coordinates, mode="bilinear", align_corners=True)
        # Output has shape [1, C, 1, num_points] so squeeze it
        F_qi_plus_di = F_qi_plus_di.squeeze(2)  # shape [1, C, num_points]

        loss += torch.nn.functional.l1_loss(F_qi.detach(), F_qi_plus_di)
    return loss, coordinates, shifted_coordinates


def feature_tracking(
        F: torch.Tensor,
        F0: torch.Tensor,
        handle_points: torch.Tensor,
        handle_points0: torch.Tensor,
        Rm,
        device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    n = handle_points.shape[0]  # Number of handle points
    new_handle_points = torch.zeros_like(handle_points)

    for i in range(n):
        # Comparing the pixel values of 4 * 4 patch centered around the handle-point, 
        # rather than individual point, this will make point tracking more accurate.
        init_coordinates = torch.nonzero(utils.create_square_mask(F.shape[2], F.shape[3], center=handle_points0[i].tolist(), radius=2).to(device))
        init_value = torch.stack([F0[:,:, coord[0], coord[1]] for coord in init_coordinates])

        super_seed = torch.nonzero(utils.create_square_mask(F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=Rm).to(device))
        patch_sets = [utils.create_square_mask(F.shape[2], F.shape[3], center=seed.tolist(), radius=2).to(device) for seed in super_seed]
        patchs = torch.stack([torch.nonzero(patchs) for patchs in patch_sets])
        patch_values = torch.stack([torch.stack([F[:,:, coord[0], coord[1]] for coord in patch]) for patch in patchs])

        # Compute the L1 distance between the patch features and the initial handle point feature
        distances = torch.norm(patch_values - init_value, p=1, dim=(1, 2, 3))

        # Find the new handle point as the one with minimum distance
        min_index = torch.argmin(distances)
        new_handle_points[i] = super_seed[min_index]

    return new_handle_points, super_seed
