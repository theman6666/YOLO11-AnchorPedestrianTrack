# -*- coding: utf-8 -*-
"""
CBAM (Convolutional Block Attention Module) implementation for YOLO11
Author: theman6666 386763479@qq.com
Date: 2025-11-24 22:22:49
LastEditors: theman6666 386763479@qq.com
LastEditTime: 2025-11-26 09:14:07
FilePath: YOLO11-AnchorPedestrianTrack/src/utils/cbam.py
Description: CBAM attention mechanism for improving YOLO11 performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out


# Small helper: insert CBAM after a block output
class CBAMWrapper(nn.Module):
    def __init__(self, module, cbam_module):
        super(CBAMWrapper, self).__init__()
        self.module = module
        self.cbam = cbam_module
        
        # 复制原模块的属性以保持兼容性
        if hasattr(module, 'f'):
            self.f = module.f
        if hasattr(module, 'i'):
            self.i = module.i
        if hasattr(module, 'type'):
            self.type = module.type

    def forward(self, x):
        x = self.module(x)
        # If module returns tuple/list (some ultralytics blocks do), handle
        if isinstance(x, (tuple, list)):
            # assume the first element is the main feature map
            feat = x[0]
            feat = self.cbam(feat)
            x = list(x)
            x[0] = feat
            return type(x)(x)
        else:
            return self.cbam(x)
