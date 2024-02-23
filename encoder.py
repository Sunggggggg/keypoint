import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(ConvBnReLU(in_channels, out_channels), 
                                  ConvBnReLU(out_channels, out_channels))

    def forward(self, x1, x2):
        '''
        x1 : Low Resolution / x2 : High Resolution
        '''
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnReLU(in_channels, out_channels),
            ConvBnReLU(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(UNet, self).__init__()

        self.inc = ConvBnReLU(in_channels, out_channels)
        self.down1 = DownConv(out_channels, out_channels*2)     #[64, 128]
        self.down2 = DownConv(out_channels*2, out_channels*4)   #[128, 256]
        self.down3 = DownConv(out_channels*4, out_channels*8)   #[256, 512]
        self.down4 = DownConv(out_channels*8, out_channels*8)   #[512, 512]

        self.up1 = UpConv(out_channels*16, out_channels*8, scale_factor=1)      #[1024, 512]
        self.up2 = UpConv(out_channels*(8+4), out_channels*4)       #[512, 128]
        self.up3 = UpConv(out_channels*(4+2), out_channels*2)       #[256, 64]
        self.up4 = UpConv(out_channels*(2+1), out_channels)         #[128, 64]
        
    def forward(self, x):
        x1 = self.inc(x)        #[B, C, H, W]
        x2 = self.down1(x1)     #[B, 2C, H//2, W//2]
        x3 = self.down2(x2)     #[B, 4C, H//4, W//4]
        x4 = self.down3(x3)     #[B, 8C, H//8, W//8]
        x5 = self.down4(x4)     #[B, 8C, H//16, W//16]
    
        x = self.up1(x5, x4)    # [B, 8C, H//8, W//8]
        x = self.up2(x, x3)     # [B, 4C, H//4, W//4]
        x = self.up3(x, x2)     # [B, 2C, W//2, W//2]
        x = self.up4(x, x1)     # [B, C, H, W]
        return x
    
if __name__ == '__main__' :
    B, C, H, W = 4, 3, 256, 256
    model = UNet(out_channels=64)
    x = torch.rand((B, C, H, W))

    print(model(x).shape)
