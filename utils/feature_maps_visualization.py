#!/usr/bin/env python

"""
This script contains functions to visualize feature maps
from convolutions layers. Inspired by:
`<https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573>`_.

Usage: type "from feature_maps_visualization" import <class>" to use one class.

Contributors: Ambroise Odonnat.
"""
import torch

import matplotlib.pyplot as plt
import torch.nn as nn

from utils.utils_ import define_device


class FeatureMaps():
    def __init__(self, model, gpu_id):

        super().__init__()
        available, self.device = define_device(gpu_id)
        self.model = model.to(self.device)

    def get_conv_layers(self):

        model_weights, conv_layers = [], []
        acc = 0

        # Recover convolutional layers and weights
        model_children = list(self.model.children())
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                acc += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            acc += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print("Total convolution layers: {}".format(acc))

        self.conv_layers = conv_layers

    def get_feature_maps(self, data):

        # Move to GPU and add batch dimension
        data = data.to(self.device).unsqueeze(0)
        outputs, names = [], []
        processed = []
        for layer in self.conv_layers[0:]:
            data = layer(data)
            outputs.append(data)
            names.append(str(layer))
        self.names = names

        # Print feature maps shapes
        for feature_map in outputs:
            print(feature_map.shape)
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        self.processed = processed

    def plot_feature_maps(self, width, height):
        fig = plt.figure(figsize=(width, height))
        for i in range(len(self.processed)):
            a = fig.add_subplot(5, 4, i+1)
            imgplot = plt.imshow(self.processed[i])
            a.axis("off")
            a.set_title(self.names[i].split('(')[0], fontsize=30)
        plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
