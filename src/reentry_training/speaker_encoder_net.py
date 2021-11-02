#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

EPS = 1e-8
import copy
def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SpeakerEncoder(nn.Module):
    def __init__(self,  B = 256, num_speaker=800):
        super(SpeakerEncoder, self).__init__()
        # self.a_encoder = audioEncoder(L=40, N=256)
        self.spk_encoder=speaker_encoder(B)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self,s):
        # s = self.a_encoder(s)
        x, x_avg = self.spk_encoder(s)
        return x, x_avg

class ResBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, out_dims, eps=1e-8)
        self.norm2 = nn.GroupNorm(1, out_dims, eps=1e-8)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.mp = nn.AvgPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.downsample:
            residual = self.conv_downsample(residual)
        x = x + residual
        x = self.prelu2(x)
        x = self.mp(x)
        return x

class speaker_encoder(nn.Module):
    def __init__(self, B, R=3, H=256):
        super(speaker_encoder, self).__init__()
        self.layer_norm = ChannelWiseLayerNorm(B)
        self.bottleneck_conv1x1 = nn.Conv1d(B, B, 1, bias=False)

        self.mynet = nn.Sequential(
            ResBlock(B, B),
            ResBlock(B, H),
            ResBlock(H, H),
            nn.Dropout(0.9),
            nn.Conv1d(H, B, 1, bias=False)
        )
        self.avgPool=nn.AdaptiveAvgPool1d(1)
        # self.dropout=nn.Dropout(0.9)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)
        x = self.mynet(x)

        x_avg = self.avgPool(x)
        return x, x_avg.squeeze(2)

class audioEncoder(nn.Module):
    def __init__(self, L, N):
        super(audioEncoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv1d_U(x))
        return x

class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x