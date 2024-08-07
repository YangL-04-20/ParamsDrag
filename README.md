![image](https://github.com/user-attachments/assets/c448ef0f-c5e3-45fa-a7f4-63ddf0c035e4)# ParamsDrag
The source code for our IEEE VIS 2024 paper "ParamsDrag: Interactive Parameter Space Exploration via Image-Space Dragging". This branch is for the Gadget2 dataset, which is 512 resolution.

>**ParamsDrag: Interactive Parameter Space Exploration via Image-Space Dragging**<br>
> Guan Li, Yang Liu, Guihua Shan, Shiyu Cheng, Weiqun Cao, Junpeng Wang, Ko-Chih Wang<br>
> [https://arxiv.org/abs/1812.04948]
>
> **Abstract:** *Numerical simulation serves as a cornerstone in scientific modeling, yet the process of fine-tuning simulation parameters poses significant challenges. Conventionally, parameter adjustment relies on extensive numerical simulations, data analysis, and expert insights, resulting in substantial computational costs and low efficiency. The emergence of deep learning in recent years has provided promising avenues for more efficient exploration of parameter spaces. However, existing approaches often lack intuitive methods for precise parameter adjustment and optimization. To tackle these challenges, we introduce ParamsDrag, a model that facilitates parameter space exploration through direct interaction with visualizations. Inspired by DragGAN, our ParamsDrag model operates in three steps. First, the generative component of ParamsDrag generates visualizations based on the input simulation parameters. Second, by directly dragging structure-related features in the visualizations, users can intuitively understand the controlling effect of different parameters. Third, with the understanding from the earlier step, users can steer ParamsDrag to produce dynamic visual outcomes. Through experiments conducted on real-world simulations and comparisons with state-of-the-art deep learning-based approaches, we demonstrate the efficacy of our solution.*

## Getting Started
### Environment Requirements
* Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
* We have done all training and development with 2 NVIDIA V100 GPUs.
* 64-bit Python 3.6 & PyTorch 1.10.0 & Torchvision 0.11.1. We recommend Anaconda3 with numpy 1.14.3 or newer.
* CUDA toolkit 11.2 or later. Use at least version 11.1 if running on RTX 3090.
* 


### Training with Generator


```
cd Model
python train --root dataset_dir \
             --dsp 3 \
             --dvp 3 \
             --d-latent 512 \
             --l1-loss \
             --perc-loss \
             --edge-loss \
             --edge-weight 0.01 \
             --batch-size 16 \
             --epochs 500 \
             --save-every 10 \
             --check-every 20
```
