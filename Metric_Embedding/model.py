import torch
import torch.nn as nn
import torch.nn.functional as F
    

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3):
        super(_BNReluConv, self).__init__()
        # self.add_module('norm', nn.GroupNorm(1, num_maps_in))  # GroupNorm with 1 group
        # self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k//2))

        self.add_module('norm', nn.GroupNorm(1, num_maps_in))  # GroupNorm with 1 group
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k//2))
        self.add_module('relu', nn.ReLU(inplace=True))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super(SimpleMetricEmbedding, self).__init__()
        self.feature_layers = nn.Sequential(
            _BNReluConv(input_channels, emb_size),
            nn.MaxPool2d(3, stride=2, padding=1),
            _BNReluConv(emb_size, emb_size),
            nn.MaxPool2d(3, stride=2, padding=1),
            _BNReluConv(emb_size, emb_size),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def get_features(self, img):
        x = self.feature_layers(img)
        x = x.view(x.size(0), -1)
        return x

    def loss(self, anchor, positive, negative, margin=5.0):
        anchor_features = self.get_features(anchor)
        positive_features = self.get_features(positive)
        negative_features = self.get_features(negative)
        
        positive_distance = F.pairwise_distance(anchor_features, positive_features, p=2)
        negative_distance = F.pairwise_distance(anchor_features, negative_features, p=2)
        
        losses = F.relu(positive_distance - negative_distance + margin)
        return losses.mean()

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # Assuming img is a PyTorch tensor of shape (batch_size, channels, height, width)
        feats = img.view(img.size(0), -1)  # <----------- Flatten the image
        return feats
