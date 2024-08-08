# Copyright 2024 The ParamsDrag Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# sparams drag
import time
import os
import sys
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import torch
import numpy as np
from drag_pipline import *

search_patch.counter = 0

device = torch.device("cuda")
G = load_model(network_pkl=resume)

# set initial parameters
sparams = torch.tensor([[0.1, 0.9, 1.0]]).to(device)
vparams = torch.tensor([[0., 0., 1.0]]).to(device)

# set handle_point & target_point
a = torch.tensor([306, 233]).float()
handle_point = (a,)
b = torch.tensor([180, 267]).float()
target_point = (b,)

orimage = G(sp=sparams, vp=vparams, force_fp32=True)

maxit = 80
switch = 1
out = drag_gan(sparams, vparams, G, handle_point, target_point, mask=None, max_iters=maxit, Rm=8, d=1, lr=0.01, bound=15, threshhold=15)
for i in range(maxit):
    img, W_out, handle_points = next(out)
    if switch == 1:
        plt.imshow(img)
    if switch == 2:
        if i == 0:
            output_path = "./"+ \
                str(np.around(np.array(sparams.tolist()[0]), 3))+'-'+ \
                str(handle_point[0].tolist())+str(target_point[0].tolist())+'-['+ resume.replace('/', '=') + ']/'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            else:
                print("************** Already exists, Please make a new dir to save ***************")
                break
        img.save(output_path + str(i)+".png")
