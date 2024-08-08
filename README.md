# ParamsDrag
![image](https://github.com/user-attachments/assets/2f2d5b84-4f6f-4ccd-a8b6-413f291ac631)

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
* The dataset consists of two folds, trian and test. labels are file names.


### Training with Generator
![image](https://github.com/user-attachments/assets/ec3a4829-043e-4872-98f0-0d39bdd3fdc7)

```
cd Model
python train.py --root dataset_dir \
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

### Drag with pkl files
![image](https://github.com/user-attachments/assets/b2a1c605-98e8-4746-8ac2-6e793ea64336)
* Running run.py could edit generative images that you have trained.
* Editable arguments include：
  * **resume**(_the model checkpoint_)
  * **sparams&vparams**(_initial parameters that you want to start from_)
  * **handle_point&target_point**(_structure you want to select_)
  * **maxit**(_maximum iterations_)
  * **bound&threshhold**(_bound is maximum structure-patch radius, threshhold is maximum flotation value of pixel, they both control the final structure-patch_)
  * **Rm**(_maximum search radius_)
  * **d**(_minimum stopping distance_)
  * **lr**(_learning rate_)


## Citation

If you use this code for your research, please cite our paper.
```
@article{ParamsDrag2024VIS,
  title={ParamsDrag: Interactive Parameter Space Exploration via Image-Space Dragging},
  author={Guan Li, Yang Liu, Guihua Shan, Shiyu Cheng, Weiqun Cao, Junpeng Wang, Ko-Chih Wang},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgments
This work was supported by the National Key Research and Development Program of China (Grant No.2023YFB3002500) and the National Natural Science Foundation of China (No.62202446). The numerical calculations in this study were carried out on the ORISE Supercomputer and the Earth System Numerical Simulation Facility (EarthLab).


