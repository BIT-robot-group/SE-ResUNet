import torch.nn as nn
import torch.nn.functional as F
import torch
from inference.models.grasp_model import GraspModel, ResidualBlock
from .senet.se_resnet import SEBasicBlock,SEBottleneck

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.up(x2)
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


class SEResUNet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(SEResUNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.seres1 = SEBottleneck(64, 64)
        self.seres2 = SEBottleneck(128, 128)
        self.seres3 = SEBottleneck(256, 256)
        self.seres4 = SEBottleneck(512, 512)

        self.max_pool1 = Down(128,256)
        self.max_pool2 = Down(256,512)

        self.up = nn.Upsample(scale_factor=2)
        factor = 2
        self.up1 = Up(1024, 512//factor)
        self.up2 = Up(512, 256//factor)
        self.up3 = Up(256, 128//factor)
        self.up4 = Up(128, 64// factor)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3,padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x0 = x
        x = F.relu(self.bn3(self.conv3(x)))
        x1 = x

        x = self.seres2(x)
        
        x = self.max_pool1(x)
        x2 = x
        x = self.seres3(x)
        x = self.max_pool2(x)
        x3 = x
        x = self.seres4(x)
        x = self.up1(x, x3)

        x = self.seres3(x)
        x = self.up2(x, x2)
    
        x = self.seres2(x)
        x = self.up3(x,x1)

        x = self.seres1(x)
        x = self.up4(x, x0)
        
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
