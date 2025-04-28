import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import math
#from .utils import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple
import torch
import torch.nn as nn
from collections import OrderedDict
import math

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseYOLO3D(nn.Module):
    def __init__(
        self,
        img_channels=1,
        out_channels=7,
        growth_rate=16,
        block_config=(2, 6, 4, 12),
        num_init_features=8,
        bn_size=4,
        drop_rate=0.0,
    ):
        super(DenseYOLO3D, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(img_channels, num_init_features, kernel_size=5,
                                        padding=2, bias=False)),
                    ("norm0", nn.BatchNorm3d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(kernel_size=2, stride=2)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm1", nn.BatchNorm3d(num_features))
        self.features.add_module(
            "conv1",
            nn.Conv3d(num_features, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False)
        )

        # Initialization
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, input_shapes):
        x = self.features(x)
        batch_size = x.shape[0]
        
        # Compute max grid sizes
        max_grid_d = max(math.ceil(D / 16) for D, _, _ in input_shapes)
        max_grid_h = max(math.ceil(H / 128) for _, H, _ in input_shapes)
        max_grid_w = max(math.ceil(W / 128) for _, _, W in input_shapes)
        
        output = []
        for i in range(batch_size):
            D, H, W = input_shapes[i]
            target_d = math.ceil(D / 16)
            target_h = math.ceil(H / 128)
            target_w = math.ceil(W / 128)
            xi = nn.functional.adaptive_avg_pool3d(x[i:i+1], (target_d, target_h, target_w))
            
            # Pad to max grid sizes
            pad_d = max_grid_d - target_d
            pad_h = max_grid_h - target_h
            pad_w = max_grid_w - target_w
            xi = torch.nn.functional.pad(
                xi, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0
            )
            output.append(xi)
        
        x = torch.cat(output, dim=0)  # [batch_size, 7, max_ceil(D/16), max_ceil(H/128), max_ceil(W/128)]
        obj = torch.sigmoid(x[:, 0:1, :, :, :])
        loc = torch.tanh(x[:, 1:4, :, :, :])
        box = torch.sigmoid(x[:, 4:7, :, :, :])
        x = torch.cat([obj, loc, box], dim=1)
        return x