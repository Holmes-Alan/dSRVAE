# dSRVAE (Generative Variational AutoEncoder for Real Image Super-Resolution)

By Zhi-Song Liu, Li-Wen Wang, Chu-Tak Li, Marie-Paule Cani and Wan-Chi Siu

This repo only provides simple testing codes, pretrained models and the network strategy demo.

We propose a joint image denoising and Super-Resolution model by using generative Variational AutoEncoder (dSRVAE)

We participate CVPRW [NTIRE2020 Real Image Super-Resolution Challenge](https://data.vision.ee.ethz.ch/cvl/ntire20/)

Please check our [paper](https://arxiv.org/abs/2004.12811)

# BibTex

        @InProceedings{Liu2020dsrvae,
            author = {Zhi-Song Liu, Wan-Chi Siu and Li-Wen Wang and Chu-Tak Li and Marie-Paule Cani and Yui-Lam Chan},
            title = {Unsupervised Real Image Super-Resolution via Generative Variational AutoEncoder},
            booktitle = {IEEE International Conference on Computer Vision and Pattern Recognition Workshop(CVPRW)},
            month = {June},
            year = {2020}
        }
        
# For proposed dSRVAE model, we claim the following points:

• First working on using Variational AutoEncoder for image denoising.

• Then the Super-Resolution Sub-Network (SRSN) is attached as a small overhead to the DAE which forms the proposed dSRVAE to output super-resolved images.

# Dependencies
    Python > 3.0
    OpenCV library
    Pytorch > 1.0
    NVIDIA GPU + CUDA
    pytorch-gan-metrics

# Complete Architecture
The complete architecture is shown as follows,

![network](/figure/figure1.png)

# Implementation
## 1. Quick testing
---------------------------------------
1. Download pre-trained models from

https://drive.google.com/open?id=1SUZGE04vw5_yDYiw6PJ4sbHAOIEV6TJ7

and copy them to the folder "models"

2. Copy your image to folder "Test" and run 
```sh
$ python test.py
```
The SR images will be in folder "Result"
3. For self-ensemble, run
```sh
$ python test_enhance.py
```
4. GAN feature evaluation
```
# download statistics.npz from http://bioinf.jku.at/research/ttur/
from pytorch_gan_metrics import get_inception_score, get_fid

images = ... # [N, 3, H, W] normalized to [0, 1]
IS, IS_std = get_inception_score(images)        # Inception Score
FID = get_fid(images, 'path/to/statistics.npz') # Frechet Inception Distance
```


## 2. Testing for NTIRE 20202
---------------------------------------

### s1. Testing images on NTIRE2020 Real World Super-Resolution Challenge - Track 1: Image Processing artifacts can be downloaded from the following link:

https://drive.google.com/open?id=10ZutE-0idGFW0KUyfZ5-2aVSiA-1qUCV

### s2. Testing images on NTIRE2020 Real World Super-Resolution Challenge - Track 2: Smartphone Images can be downloaded from the following link:

https://drive.google.com/open?id=1_R4kRO_029g-HNAzPobo4-xwp86bMZLW

### s3. Validation images on NTIRE2020 Real World Super-Resolution Challenge - Track 1 and Track 2 can be downloaded from the following link:

https://drive.google.com/open?id=1nKEJ4N2V-0NFicfJxm8AJqsjXoGMYjMp

## 3. Training
---------------------------
### s1. Download the training images from NTIRE2020.
    
https://competitions.codalab.org/competitions/22220#learn_the_details

   
### s2. Start training on Pytorch
1. Train the Denoising VAE by running
```sh
$ python main_denoiser.py
```
2. Train the super-resolution SRSN overhead by running
```sh
$ python main_GAN.py
```
---------------------------

## Partial image visual comparison

## 1. Visualization comparison
Results on 4x image SR on Track 1 dataset
![figure2](/figure/figure2.png)
![figure3](/figure/figure3.png)
![figure4](/figure/figure4.png)


# Reference
You may check our newly work on [General image super-resolution using VAE](https://github.com/Holmes-Alan/SR-VAE)

You may also check our work on [Reference based face SR using VAE](https://github.com/Holmes-Alan/RefSR)

You may also check our work on [Reference based General image SR using VAE](https://github.com/Holmes-Alan/RefVAE)

Special thanks to the contributions of Jakub M. Tomczak for their [VAE with a VampPrior](https://github.com/jmtomczak/vae_vampprior) on KL loss calculation. 
