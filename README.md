# SRVAE (Generative Variational AutoEncoder for Real Image Super-Resolution)

By Zhi-Song, Li-Wen Wang, Marie-Paule Cani and Wan-Chi Siu

This repo only provides simple testing codes, pretrained models and the network strategy demo.

We propose a joint image denoising and Super-Resolution model by using generative Variational AutoEncoder (dSRVAE)

# For proposed dSRVAE model, we claim the following points:

• First working on using Variational AutoEncoder for image denoising.

• Then the Super-Resolution Sub-Network (SRSN) is attached as a small overhead to the DAE which forms the proposed dSRVAE to output super-resolved images.

# Dependencies
    Python > 3.0
    OpenCV library
    Pytorch > 1.2 
    NVIDIA GPU + CUDA

# Complete Architecture
The complete architecture is shown as follows,

![network](/figure/network.png)

# Implementation
## 1. Quick testing
---------------------------------------
### Copy your image to folder "Test" and run test.sh. The SR images will be in folder "Result"

## 2. Testing for NTIRE 20202
---------------------------------------

### s1. Testing images on NTIRE2020 Extreme Super-Resolution Challenge - Track 1: Image Processing artifacts can be downloaded from the following link:

https://competitions.codalab.org/competitions/20235

### s2. Testing images on AIM2019 Constrained Super-Resolution Challenge - Track 3: Fidelity optimization can be downloaded from the following link:

https://competitions.codalab.org/competitions/20169

General testing dataset (Set5, Set14, BSD100, Urban100 and Manga109) can be downloaded from:

https://github.com/LimBee/NTIRE2017

## 3. Training
---------------------------
### s1. Download the training images from NTIRE2020.
    
https://competitions.codalab.org/competitions/22220#learn_the_details

   
### s2. Start training on Pytorch
For user who already has installed Pytorch 1.1, simply just run the following code for AIM2019 Constrained Super-Resolution Challenge - Track 3: Fidelity optimization:
```sh
$ python main_GAN.py
```
---------------------------

## Partial image visual comparison

## 1. Visualization comparison
Results on 4x image SR on Track 1 dataset
![visual](/figure/visualization.png)
