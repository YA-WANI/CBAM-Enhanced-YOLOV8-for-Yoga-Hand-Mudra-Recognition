import os
import yaml
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, SPPF, Conv, SiLU, BatchNorm2d
from ultralytics.nn.tasks import Detect, DetectionModel
from torch.nn import Sequential, Conv2d

# 3. Disable deterministic behavior
torch.use_deterministic_algorithms(False)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


# 5. Define CBAM module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(1, in_planes // ratio)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, reduced_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# 6. Patch torch.load to support custom CBAM layer
import torch.serialization
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    Sequential,
    Conv,
    Conv2d,
    SiLU,
    BatchNorm2d,
    C2f,
    SPPF,
    CBAM
])

# 7. Register CBAM into Ultralytics
from ultralytics.nn import modules as ultralytics_modules
ultralytics_modules.C2f = C2f
ultralytics_modules.SPPF = SPPF
ultralytics_modules.Detect = Detect
ultralytics_modules.CBAM = CBAM
ultralytics.nn.tasks.__dict__['CBAM'] = CBAM

# 8. Get width and depth multipliers per variant
def get_variant_hyperparams(variant):
    if variant == 'n': return 0.33, 0.25
    if variant == 's': return 0.33, 0.50
    if variant == 'm': return 0.67, 0.75
    if variant == 'l': return 1.00, 1.00
    if variant == 'x': return 1.00, 1.25
    raise ValueError("Invalid variant.")

# 9. YAML builder
def get_dynamic_yaml(base_conv_channels_unscaled, variant='s'):
    dm, wm = get_variant_hyperparams(variant)

    c1_base = base_conv_channels_unscaled
    c2_base = int(c1_base * 2)
    c3_base = int(c1_base * 4)
    c4_base = int(c1_base * 8)

    actual_c1 = max(8, int(c1_base * wm))
    actual_c2 = max(8, int(c2_base * wm))
    actual_c3 = max(8, int(c3_base * wm))
    actual_c4 = max(8, int(c4_base * wm))

    c1_base = int(actual_c1 / wm)
    c2_base = int(actual_c2 / wm)
    c3_base = int(actual_c3 / wm)
    c4_base = int(actual_c4 / wm)

    n_c2f = max(1, int(3 * dm))

    return f"""
nc: 10
depth_multiple: {dm}
width_multiple: {wm}

backbone:
  [
    [-1, 1, Conv, [{c1_base}, 3, 2]],
    [-1, 1, Conv, [{c2_base}, 3, 2]],  # No CBAM in early stage
    [-1, {n_c2f}, C2f, [{c2_base}, True]],
    [-1, 1, Conv, [{c3_base}, 3, 2]],
    [-1, {n_c2f}, C2f, [{c3_base}, True]],
    [-1, 1, CBAM, [{actual_c3}]],      # CBAM in deeper layers only
    [-1, 1, Conv, [{c4_base}, 3, 2]],
    [-1, 1, SPPF, [{c4_base}, 5]],
    [-1, 1, CBAM, [{actual_c4}]]       # CBAM after SPPF
  ]

head:
  [
    [-1, 1, Conv, [{c3_base}, 1, 1]],
    [-1, 1, CBAM, [{actual_c3}]],
    [[[-1], 1, Detect, [nc]]]
  ]
"""
