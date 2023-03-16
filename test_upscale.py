import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        dims = [32,16]
        dimsx = [64,32,16]
        for i in range(len(dims)):
            downsample = nn.Conv2d(dimsx[i], dimsx[i+1], kernel_size=3, stride=2, padding=1)
            upsample = nn.ConvTranspose2d(dimsx[-i-1], dimsx[-i-2], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.downscale.append(downsample)
            self.upscale.append(upsample)

    def forward(self, x):
        x = self.conv1(x)
        body_out = x
        for layer in self.downscale:
            x = layer(x)
        for layer in self.upscale:
            x = layer(x)
        x = torch.cat((x, body_out), dim=1)
        return x

model = CNN()
model.eval()
input = torch.rand((1, 3, 512, 512))
print(summary(model, input, show_input=False, show_hierarchical=False))