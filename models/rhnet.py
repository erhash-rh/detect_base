import torch
import torch.nn as nn
import torch.nn.functional as F

class rHnetFPN(nn.Module):
    def __init__(self, depths=[2, 2], stem_dims=[64, 64], dims=[64, 128], fpn_dims=[64,64,64], in_channels=3, class_dims=[64,32], expand=2, classes=0):
        super().__init__()

        # STEM
        self.stem = nn.ModuleList()
        stem_dimsx = [in_channels] + stem_dims
        for i in range(len(stem_dimsx)-1):
            layer = nn.Sequential(nn.Conv2d(stem_dimsx[i], stem_dimsx[i+1], kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(stem_dimsx[i+1]))
            self.stem.append(layer)
        
        # BODY
        self.body = nn.ModuleList()
        dimsx = [stem_dims[-1]] + dims
        for i in range(len(depths)):
            section = nn.Sequential(*[Block(dimsx[i], expand=expand) for _ in range(depths[i])], nn.Conv2d(dimsx[i], dimsx[i+1], kernel_size=3, stride=2, padding=1), 
                                    nn.ReLU(), nn.BatchNorm2d(dimsx[i+1]))
            self.body.append(section)
        
        # FPN
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        fpn_dimsx = [dims[-1]] + fpn_dims
        for i in range(len(fpn_dims)):
            downsample = nn.Sequential(nn.Conv2d(fpn_dimsx[i], fpn_dimsx[i+1], kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(fpn_dimsx[i+1]))
            upsample = nn.Sequential(nn.ConvTranspose2d(fpn_dimsx[-i-1], fpn_dimsx[-i-2], kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(fpn_dimsx[-i-2]))
            self.downscale.append(downsample)
            self.upscale.append(upsample)
        
        # HEAD
        self.head1 = nn.ModuleList()
        self.head2 = nn.ModuleList()
        self.head3 = nn.ModuleList()
        class_dimsx = [dims[-1]] + class_dims
        for i in range(len(class_dimsx)-1):
            self.head1.append(nn.Sequential(nn.Conv2d(class_dimsx[i], class_dimsx[i+1], kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(class_dimsx[i+1])))
            self.head2.append(nn.Sequential(nn.Conv2d(class_dimsx[i], class_dimsx[i+1], kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(class_dimsx[i+1])))
            self.head3.append(nn.Sequential(nn.Conv2d(class_dimsx[i], class_dimsx[i+1], kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(class_dimsx[i+1])))
        self.head1.append(nn.Conv2d(class_dimsx[-1], 1, kernel_size=3, stride=1, padding=1))
        self.head2.append(nn.Conv2d(class_dimsx[-1], 2, kernel_size=3, stride=1, padding=1))
        self.head3.append(nn.Conv2d(class_dimsx[-1], 2, kernel_size=3, stride=1, padding=1))

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2/n)**0.5)
    
    def forward(self, x):
        # STEM
        for i in range(len(self.stem)):
            x = self.stem[i](x)

        # BODY
        for i in range(len(self.body)):
            x = self.body[i](x)

        # FPN
        body_out = x
        for layer in self.downscale:
            x = layer(x)
        for layer in self.upscale:
            x = layer(x)
        x = torch.add(body_out, x, alpha=0.5)

        # HEAD
        x2=x
        x3=x
        for i in range(len(self.head1)):
            x = self.head1[i](x)
            x2 = self.head2[i](x2)
            x3 = self.head2[i](x3)
        
        x = torch.cat((x,x2,x3), dim=1)

        return x


class rHnet(nn.Module):
    def __init__(self, depths=[2, 2], stem_dims=[64, 64], dims=[64, 128], in_channels=3, class_dims=[64,32], expand=2, classes=0):
        super().__init__()

        # STEM
        self.stem = nn.ModuleList()
        stem_dimsx = [in_channels] + stem_dims
        for i in range(len(stem_dimsx)-1):
            layer = nn.Sequential(nn.Conv2d(stem_dimsx[i], stem_dimsx[i+1], kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(stem_dimsx[i+1]))
            self.stem.append(layer)
        
        # BODY
        self.body = nn.ModuleList()
        dimsx = [stem_dims[-1]] + dims
        for i in range(len(depths)):
            section = nn.Sequential(*[Block(dimsx[i], expand=expand) for _ in range(depths[i])], nn.Conv2d(dimsx[i], dimsx[i+1], kernel_size=3, stride=2, padding=1), 
                                    nn.ReLU(), nn.BatchNorm2d(dimsx[i+1]))
            self.body.append(section)
        
        # HEAD
        self.head1 = nn.ModuleList()
        self.head2 = nn.ModuleList()
        self.head3 = nn.ModuleList()
        class_dimsx = [dims[-1]] + class_dims
        for i in range(len(class_dimsx)-1):
            self.head1.append(nn.Sequential(nn.Conv2d(class_dimsx[i], class_dimsx[i+1], kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(class_dimsx[i+1])))
            self.head2.append(nn.Sequential(nn.Conv2d(class_dimsx[i], class_dimsx[i+1], kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(class_dimsx[i+1])))
            self.head3.append(nn.Sequential(nn.Conv2d(class_dimsx[i], class_dimsx[i+1], kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(class_dimsx[i+1])))
        self.head1.append(nn.Conv2d(class_dimsx[-1], 1, kernel_size=3, stride=1, padding=1))
        self.head2.append(nn.Conv2d(class_dimsx[-1], 2, kernel_size=3, stride=1, padding=1))
        self.head3.append(nn.Conv2d(class_dimsx[-1], 2, kernel_size=3, stride=1, padding=1))

        #self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2/n)**0.5)
    
    def forward(self, x):
        for i in range(len(self.stem)):
            x = self.stem[i](x)
        for i in range(len(self.body)):
            x = self.body[i](x)
        x2=x
        x3=x
        for i in range(len(self.head1)):
            x = self.head1[i](x)
            x2 = self.head2[i](x2)
            x3 = self.head3[i](x3)
        
        x = torch.cat((x,x2,x3), dim=1)

        return x
        
class Block(nn.Module):
    def __init__(self, channels, expand):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels*expand, kernel_size=3, stride=1, padding=1)
        self.a1 = nn.ReLU()
        self.c2 = nn.Conv2d(channels*expand, channels, kernel_size=1, stride=1)
        self.a2 = nn.ReLU()
    
    def forward(self, x):
        input = x
        x = self.c1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.a2(x)
        x = input + x
        return x