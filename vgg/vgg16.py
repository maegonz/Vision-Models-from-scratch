import torch.nn as nn
from .convblock import ConvBlock

class VGG16(nn.Module):
    def __init__(self, num_classes:int, dropout:float=0.0, batch_normalization:bool=False):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(num_convs=2, in_channels=3, out_channels=64, dropout=dropout, batch_normalization=batch_normalization),
            ConvBlock(num_convs=2, in_channels=64, out_channels=128, dropout=dropout, batch_normalization=batch_normalization),
            ConvBlock(num_convs=3, in_channels=128, out_channels=256, dropout=dropout, batch_normalization=batch_normalization),
            ConvBlock(num_convs=3, in_channels=256, out_channels=512, dropout=dropout, batch_normalization=batch_normalization),
            ConvBlock(num_convs=3, in_channels=512, out_channels=512, dropout=dropout, batch_normalization=batch_normalization)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = nn.Softmax(dim=1)(x)
        return x