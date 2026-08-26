import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):  # 기본 32 (기존 64 절반)
        super().__init__()
        self.dconv1 = DoubleConv(in_ch, base_ch)
        self.dconv2 = DoubleConv(base_ch, base_ch*2)
        self.dconv3 = DoubleConv(base_ch*2, base_ch*4)
        self.dconv4 = DoubleConv(base_ch*4, base_ch*8)
        self.dconv5 = DoubleConv(base_ch*8, base_ch*16)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.uconv4 = DoubleConv(base_ch*8+base_ch*16, base_ch*8)
        self.uconv3 = DoubleConv(base_ch*4+base_ch*8, base_ch*4)
        self.uconv2 = DoubleConv(base_ch*2+base_ch*4, base_ch*2)
        self.uconv1 = DoubleConv(base_ch+base_ch*2, base_ch)

        self.out = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        c1 = self.dconv1(x)
        c2 = self.dconv2(self.pool(c1))
        c3 = self.dconv3(self.pool(c2))
        c4 = self.dconv4(self.pool(c3))
        c5 = self.dconv5(self.pool(c4))

        u4 = self.up(c5)
        u4 = torch.cat([u4, c4], dim=1)
        u4 = self.uconv4(u4)

        u3 = self.up(u4)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.uconv3(u3)

        u2 = self.up(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.uconv2(u2)

        u1 = self.up(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.uconv1(u1)

        return self.out(u1)
