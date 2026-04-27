import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(num_channels: int, num_groups: int = 8):
    g = min(num_groups, num_channels)
    return nn.GroupNorm(g, num_channels)


class SegHeadFromResNetFeatures(nn.Module):
    """
    U-Net decoder that handles non-divisible spatial dimensions.
    """
    def __init__(self, base=32, num_classes=1, dropout=0.0):
        super().__init__()
        self.base = base
        
        # Upsample + conv blocks
        self.up4 = self._up_block(base * 8, base * 4)
        self.conv4 = self._conv_block(base * 4 + base * 4, base * 4)
        
        self.up3 = self._up_block(base * 4, base * 2)
        self.conv3 = self._conv_block(base * 2 + base * 2, base * 2)
        
        self.up2 = self._up_block(base * 2, base)
        self.conv2 = self._conv_block(base + base, base)
        
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.final = nn.Conv3d(base, num_classes, kernel_size=1)
        
    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2),
            gn(out_c),
            nn.ReLU(inplace=True),
        )
    
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            gn(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            gn(out_c),
            nn.ReLU(inplace=True),
        )
    
    def _match_size(self, x, target):
        """
        Resize x to match target's spatial dimensions.
        Handles cases where upsampling doesn't exactly match skip connection size.
        """
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)
        return x
    
    def forward(self, skips):
        """
        Args:
            skips: dict with keys 'layer1', 'layer2', 'layer3', 'layer4'
        """
        x = skips['layer4']
        
        x = self.up4(x)
        x = self._match_size(x, skips['layer3'])
        x = torch.cat([x, skips['layer3']], dim=1)
        x = self.conv4(x)
        
        x = self.up3(x)
        x = self._match_size(x, skips['layer2'])
        x = torch.cat([x, skips['layer2']], dim=1)
        x = self.conv3(x)
        
        x = self.up2(x)
        x = self._match_size(x, skips['layer1'])
        x = torch.cat([x, skips['layer1']], dim=1)
        x = self.conv2(x)
        
        x = self.dropout(x)
        x = self.final(x)
        
        return x