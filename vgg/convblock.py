import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, num_convs:int, in_channels:int, out_channels:int, dropout:float=0.0, batch_normalization:bool=False):
        super(ConvBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_convs):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_normalization:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x