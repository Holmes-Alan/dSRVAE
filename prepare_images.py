import copy


import torch.nn as nn
import torch


class ImageSplitter:
    # key points:
    # Boarder padding and over-lapping img splitting to avoid the instability of edge value
    # Thanks Waifu2x's autorh nagadomi for suggestions (https://github.com/nagadomi/waifu2x/issues/238)

    def __init__(self, patch_size, scale_factor, stride):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.stride = stride
        self.height = 0
        self.width = 0

    def split_img_tensor(self, img_tensor):
        # resize image and convert them into tensor
        batch, channel, height, width = img_tensor.size()
        self.height = height
        self.width = width

        side = min(height, width, self.patch_size)
        delta = self.patch_size - side
        Z = torch.zeros([batch, channel, height+delta, width+delta])
        Z[:, :, delta//2:height+delta//2, delta//2:width+delta//2] = img_tensor
        batch, channel, new_height, new_width = Z.size()

        patch_box = []

        # split image into over-lapping pieces
        for i in range(0, new_height, self.stride):
            for j in range(0, new_width, self.stride):
                x = min(new_height, i + self.patch_size)
                y = min(new_width, j + self.patch_size)
                part = Z[:, :, x-self.patch_size:x, y-self.patch_size:y]

                patch_box.append(part)

        patch_tensor = torch.cat(patch_box, dim=0)
        return patch_tensor

    def merge_img_tensor(self, list_img_tensor):
        img_tensors = copy.copy(list_img_tensor)

        patch_size = self.patch_size * self.scale_factor
        stride = self.stride * self.scale_factor
        height = self.height * self.scale_factor
        width = self.width * self.scale_factor
        side = min(height, width, patch_size)
        delta = patch_size - side
        new_height = delta + height
        new_width = delta + width
        out = torch.zeros((1, 3, new_height, new_width))
        mask = torch.zeros((1, 3, new_height, new_width))

        for i in range(0, new_height, stride):
            for j in range(0, new_width, stride):
                x = min(new_height, i + patch_size)
                y = min(new_width, j + patch_size)
                mask_patch = torch.zeros((1, 3, new_height, new_width))
                out_patch = torch.zeros((1, 3, new_height, new_width))
                mask_patch[:, :, (x - patch_size):x, (y - patch_size):y] = 1.0
                out_patch[:, :, (x - patch_size):x, (y - patch_size):y] = img_tensors.pop(0)
                mask = mask + mask_patch
                out = out + out_patch

        out = out / mask

        out = out[:, :, delta//2:new_height - delta//2, delta//2:new_width - delta//2]

        return out

