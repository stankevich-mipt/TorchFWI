import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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


class UNet(nn.Module):

    def __init__(self, dim1=16, dim2=32, dim3=64, dim4=128, dim5=256):
        
        super(UNet, self).__init__()

        self.inc   = DoubleConv(1, dim1)
        self.down1 = Down(dim1, dim2)
        self.down2 = Down(dim2, dim3)
        self.down3 = Down(dim3, dim4)
        self.down4 = Down(dim4, dim5)

        self.up1   = Up(dim5, dim4)
        self.up2   = Up(dim4, dim3)
        self.up3   = Up(dim3, dim2)
        self.up4   = Up(dim2, dim1)
        self.outc  = OutConv(dim1, 1)

        convblock1   = ConvBlock(1, dim1, kernel_size=(11, 7), stride=(8, 2), padding=(5, 3))
        convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(7, 1), stride=(4, 1), padding=(3, 0))
        convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        convblock3_2 = ConvBlock(dim3, dim3)
        convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        convblock4_2 = ConvBlock(dim4, dim4)
        convblock5_1 = ConvBlock(dim4, dim5, stride=2)
        convblock5_2 = ConvBlock(dim5, dim5)
    

        self.head_signal = nn.Sequential(
            convblock1,
            convblock2_1,
            convblock2_2
        )

        self.downs1 = nn.Sequential(
            convblock3_1,
            convblock3_2
        )

        self.downs2 = nn.Sequential(
            convblock4_1, 
            convblock4_2
        )

        self.downs3 = nn.Sequential(
            convblock5_1,
            convblock5_2 
        )

        self.mixer1 = ConvBlock(2 * dim2, dim2)
        self.mixer2 = ConvBlock(2 * dim3, dim3)
        self.mixer3 = ConvBlock(2 * dim4, dim4)
        self.mixer4 = ConvBlock(2 * dim5, dim5)


    def forward(self, signal, guess):

        signal = F.pad(signal, [10, 11, 152, 152], mode='constant', value=0)
        guess  = F.pad(guess , [8, 8, 5, 5], mode='constant', value=0)

        x1     = self.inc(guess)
        signal = self.head_signal(signal)
        
        x2     = self.down1(x1)
        x2     = self.mixer1(torch.cat([signal, x2], dim=1))
        signal = self.downs1(signal)

        x3     = self.down2(x2)
        x3     = self.mixer2(torch.cat([signal, x3], dim=1))
        signal = self.downs2(signal)

        x4     = self.down3(x3)
        x4     = self.mixer3(torch.cat([signal, x4], dim=1))
        signal = self.downs3(signal)

        x5     = self.down4(x4)
        x5     = self.mixer4(torch.cat([signal, x5], dim=1))
        
        x      = self.up1(x5, x4)
        x      = self.up2(x, x3)
        x      = self.up3(x, x2)
        x      = self.up4(x, x1)
       
        out    = torch.tanh(self.outc(x))
        out    = F.pad(out, [-8, -8, -5, -5], mode='constant', value=0)

        return out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)