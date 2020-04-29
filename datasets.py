import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def modcrop(im, modulo):
    (h, w) = im.size
    new_h = h//modulo*modulo
    new_w = w//modulo*modulo
    ih = h - new_h
    iw = w - new_w
    ims = im.crop((0, 0, h - ih, w - iw))
    return ims

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, img_ref, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    #(th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - tp + 1)
    if iy == -1:
        iy = random.randrange(0, ih - tp + 1)

    (tx, ty) = (scale * ix, scale * iy)

    out_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    out_tar = img_tar.crop((iy, ix, iy + ip, ix + ip))
    out_ref = img_ref.crop((iy, ix, iy + tp, ix + tp))
    #img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    #info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return out_in, out_tar, out_ref


def augment(img_in, img_tar, img_ref, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_ref = ImageOps.flip(img_ref)
        #img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_ref = ImageOps.mirror(img_ref)
            #img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_ref = img_ref.rotate(180)
            #img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, img_ref, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)] # uncomment it
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target = load_img(self.hr_image_filenames[index])
        target = modcrop(target, 16)
        ref = load_img(self.hr_image_filenames[len(self.hr_image_filenames)-index])
        ref = modcrop(ref, 16)
        name = self.hr_image_filenames[index]
        input = load_img(self.lr_image_filenames[index])

        input, target, ref = get_patch(input, target, ref, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            input, target, ref, _ = augment(input, target, ref)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
            ref = self.transform(ref)

        return input, target, ref

    def __len__(self):
        return len(self.hr_image_filenames) # modify the hr to lr


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            #input = self.transform(input)
            bicubic = self.transform(bicubic)

        return bicubic, file

    def __len__(self):
        return len(self.image_filenames)
