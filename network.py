# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from collections import OrderedDict

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

# Replace the key names in the checkpoint in which legacy network building blocks are used 
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
              .replace('Conv2DwithBN_Tanh', 'layers')
              .replace('Deconv2DwithBN', 'layers')
              .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=0.2, dropout=None):
        super(Conv2DwithBN,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)
 
class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(Conv2DwithBN_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm=None, relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm=None):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm=None):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class InversionNet(nn.Module):
    def __init__(self, dim1=16, dim2=32, dim3=64, dim4=128, dim5=256, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()


        convblock1   = ConvBlock(1, dim1, kernel_size=(7, 7), stride=(4, 2), padding=(3, 3))
        convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        convblock5_2 = ConvBlock(dim3, dim3)
        convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        convblock6_2 = ConvBlock(dim4, dim4)
        convblock7_1 = ConvBlock(dim4, dim5, stride=2)
        convblock7_2 = ConvBlock(dim5, dim5)
        convblock8   = ConvBlock(dim5, dim5, kernel_size=(8, 25), padding=0)

        self.head_signal = nn.Sequential(
            convblock1,
            convblock2_1,
            convblock2_2,
            convblock3_1,
            convblock3_2, 
            convblock4_1, 
            convblock4_2,
            convblock5_1,
            convblock5_2, 
            convblock6_1, 
            convblock6_2,
            convblock7_1, 
            convblock7_2, 
            convblock8
        )

        convblock1   = ConvBlock(1, dim1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)) # 32 x 64 x 64
        convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=1, padding=1) # 64 x 64 x 64
        convblock2_2 = ConvBlock(dim2, dim2, kernel_size=2, stride=2, padding=0) # 64 x 32 x 32
        convblock3_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=1, padding=1) # 128 x 32 x 32
        convblock3_2 = ConvBlock(dim3, dim3, kernel_size=2, stride=2, padding=0) # 128 x 16 x 16 
        convblock4_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=1, padding=1) # 256 x 16 x 16
        convblock4_2 = ConvBlock(dim4, dim4, kernel_size=2, stride=2, padding=0) # 256 x 8 x 8
        convblock5_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=1, padding=1) # 512 x 8 x 8
        convblock5_2 = ConvBlock(dim4, dim5, kernel_size=2, stride=2, padding=0) # 512 x 4 x 4
        convblock6   = ConvBlock(dim5, dim5, kernel_size=(9, 25), padding=0)

        self.head_guess = nn.Sequential(
            convblock1,
            convblock2_1,
            convblock2_2,
            convblock3_1,
            convblock3_2, 
            convblock4_1, 
            convblock4_2,
            convblock5_1,
            convblock5_2,
            convblock6
        )

        deconv1_1 = DeconvBlock(2 * dim5, dim5, kernel_size=(9, 25))
        deconv1_2 = ConvBlock(dim5, dim5)
        deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        deconv2_2 = ConvBlock(dim4, dim4)
        deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        deconv3_2 = ConvBlock(dim3, dim3)
        deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        deconv4_2 = ConvBlock(dim2, dim2)
        deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        deconv5_2 = ConvBlock(dim1, dim1)
        deconv6   = ConvBlock_Tanh(dim1, 1)

        self.tail = nn.Sequential(
            deconv1_1,
            deconv1_2,
            deconv2_1,
            deconv2_2,
            deconv3_1,
            deconv3_2,
            deconv4_1,
            deconv4_2,
            deconv5_1,
            deconv5_2,
            deconv6
        )
        
    def forward(self, signal, guess):

        signal = F.pad(signal, [10, 11, 12, 12], mode='constant', value=0)
        guess  = F.pad(guess, [8, 8, 5, 5], mode='constant', value=0)
        signal = self.head_signal(signal)
        guess  = self.head_guess(guess)

        features = torch.cat([signal, guess], dim=1)

        pred     = self.tail(features) 
        pred     = F.pad(pred, [-8, -8, -5, -5], mode="constant", value=0)

        return pred

class FCN4_Deep_Resize_2(nn.Module):
    def __init__(self, dim1=16, dim2=32, dim3=64, dim4=128, dim5=256, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2, self).__init__()
        self.convblock1 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(70 * ratio / 8)), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class Discriminator(nn.Module):

    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):

        super(Discriminator, self).__init__()
        
        convblock1   = ConvBlock(1, dim1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)) # 32 x 64 x 64
        convblock2_1 = ConvBlock(dim1, dim2, kernel_size=2, stride=2, padding=0) # 64 x 32 x 32
        convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, stride=1, padding=1) # 64 x 64 x 64
        convblock3_1 = ConvBlock(dim2, dim3, kernel_size=2, stride=2, padding=0) # 128 x 16 x 16 
        convblock3_2 = ConvBlock(dim3, dim3, kernel_size=3, stride=1, padding=1) # 128 x 32 x 32
        convblock4_1 = ConvBlock(dim3, dim4, kernel_size=2, stride=2, padding=0) # 256 x 8 x 8
        convblock4_2 = ConvBlock(dim4, dim4, kernel_size=3, stride=1, padding=1) # 256 x 16 x 16
        convblock5_1 = ConvBlock(dim4, dim5, kernel_size=2, stride=2, padding=0) # 512 x 4 x 4
        convblock5_2 = ConvBlock(dim5, dim5, kernel_size=3, stride=1, padding=1) # 512 x 8 x 8
        convblock6   = ConvBlock(dim5, dim5, kernel_size=(9, 25), padding=0)

        self.feature_extractor = nn.Sequential(
            convblock1,
            convblock2_1,
            convblock2_2,
            convblock3_1,
            convblock3_2, 
            convblock4_1, 
            convblock4_2,
            convblock5_1,
            convblock5_2,
            convblock6
        )

        self.predictor = nn.Linear(dim5, 1)


    def forward(self, x):

        x  = F.pad(x, [8, 8, 5, 5], mode='constant', value=0)
        
        return self.predictor(self.feature_extractor(x).view(x.shape[0], -1)).view(-1)


class Conv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=None, stride=None, padding=None, **kwargs):
        super(Conv_HPGNN, self).__init__()
        layers = [
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8),
        ]
        if kernel_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Deconv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size, **kwargs):
        super(Deconv_HPGNN, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_fea, in_fea, kernel_size=kernel_size, stride=2, padding=0),
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


model_dict = {
    'InversionNet': InversionNet,
    'Discriminator': Discriminator,
    'UPFWI': FCN4_Deep_Resize_2
}

