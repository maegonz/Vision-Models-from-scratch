import torch
import torch.nn as nn
from .convblock import DoubleConv
from .layers import Encoder, Decoder

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature: int=64):
        super(UNet, self).__init__()
        self.encoder1 = Encoder(in_channels, feature)
        self.encoder2 = Encoder(feature, feature * 2)
        self.encoder3 = Encoder(feature * 2, feature * 4)
        self.encoder4 = Encoder(feature * 4, feature * 8)

        self.bottleneck = DoubleConv(feature * 8, feature * 16)

        self.decoder4 = Decoder(feature * 16, feature * 8)
        self.decoder3 = Decoder(feature * 8, feature * 4)
        self.decoder2 = Decoder(feature * 4, feature * 2)
        self.decoder1 = Decoder(feature * 2, feature)

        self.out = nn.Conv2d(feature, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1, p1 = self.encoder1(x)
        enc2, p2 = self.encoder2(p1)
        enc3, p3 = self.encoder3(p2)
        enc4, p4 = self.encoder4(p3)

        bn = self.bottleneck(p4)

        dec4 = self.decoder4(bn, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        return self.out(dec1)