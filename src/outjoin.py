import torch
import pandas as pd
import pickle
import time
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF

def crop_and_pad_images(images, crop_top=4, crop_bottom=4, crop_left=4, crop_right=4):
    """
    裁剪图像的顶部、底部、左边和右边，并用 0 填充保持尺寸一致。
    :param images: 输入的图像张量，形状为 (batch_size, channels, height, width)
    :param crop_top: 裁剪顶部的行数
    :param crop_bottom: 裁剪底部的行数
    :param crop_left: 裁剪左边的列数
    :param crop_right: 裁剪右边的列数
    :return: 裁剪并零填充后的图像张量
    """
    batch_size, channels, height, width = images.shape
    # 裁剪图像
    cropped_images = images[:, :, crop_top:height-crop_bottom, crop_left:width-crop_right]
    # 创建一个全零张量用于填充
    padded_images = torch.zeros_like(images)
    # 将裁剪后的图像放置回填充张量的中心位置
    padded_images[:, :, crop_top:height-crop_bottom, crop_left:width-crop_right] = cropped_images
    return padded_images

def rotate_images(images, angle):
    """
    旋转图像张量
    :param images: 输入的图像张量，形状为 (batch_size, channels, height, width)
    :param angle: 旋转角度（以度为单位，顺时针方向为正）
    :return: 旋转后的图像张量
    """
    rotated_images = []
    for img in images:
        # 每张图片旋转
        rotated_img = TF.rotate(img, angle)
        rotated_images.append(rotated_img)
    # 将旋转后的图像列表堆叠成张量
    return torch.stack(rotated_images)