import torch
import torch.nn as nn
import torch.nn.functional as F

def gn(num_channels: int, num_groups: int = 8):
    g = min(num_groups, num_channels)
    return nn.GroupNorm(g, num_channels)


class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.norm1 = gn(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.norm2 = gn(out_c)

        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False),
                gn(out_c),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = F.relu(out + identity, inplace=True)
        return out


class ResNet3D(nn.Module):
    def __init__(self, layers=(2,2,2,2), in_ch=1, base=32, emb_dim=256, with_skip=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, stride=1, padding=1, bias=False),
            gn(base),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base, base, layers[0], stride=1)
        self.layer2 = self._make_layer(base, base*2, layers[1], stride=2)
        self.layer3 = self._make_layer(base*2, base*4, layers[2], stride=2)
        self.layer4 = self._make_layer(base*4, base*8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base*8, emb_dim)
        self.with_skip = with_skip

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock3D(in_c, out_c, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        # returns spatial feature map: (B, C, 5, 5, 5)
        x = self.stem(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
                                                                                                                                                               
        if self.with_skip:
            print("With skip connections from ResNet3D")
            return {
                "layer1": s1,
                "layer2": s2,
                "layer3": s3,
                "layer4": s4,
                "x": x
            }
        else:
            print("Without skip connections from ResNet3D")
            return s4

    def forward(self, x):
        f = self.forward_features(x)
        x = self.pool(f).flatten(1)
        z = self.fc(x)
        return z


def resnet18_3d_gn(emb_dim=256):
    return ResNet3D(layers=(2,2,2,2), in_ch=1, base=32, emb_dim=emb_dim)

