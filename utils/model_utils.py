"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1

A library of model classes that can used in Self-Supervised framework. The list of models:

BYOL:          High-level model that contains Online and Target networks as well as Predictor network.
EncodeProject: The model class that contain Encoder and Projection networks
CNNEncoder:    CNN-based encoder used to learn useful representations.
Projector:     MLP network used to project representations.
Predictor:     MLP network used to make predictions for the output of Target network.
"""

import copy
import torch as th
from torch import nn
import torch.nn.functional as F
from torchvision import models


class BYOL(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: tuple (q, z): L2 normalized predictions (q) of online network and projection (z) of target network, both
                           of which have a shape of (batch size, feature dim)
    """
    def __init__(self, options):
        super(BYOL, self).__init__()
        # Get configuration
        self.options = options
        # Initialize online network
        self.online_network = EncodeProject(options)
        # Initialize target network
        self.target_network = EncodeProject(options)
        # Initialize Prediction network
        self.predictor = Predictor(options)

    def forward(self, x):
        # Generate predictions after projection from online network
        q = self.predictor(self.online_network(x))
        # Generate projection from target network
        with th.no_grad():
            z = self.target_network(x)
        # L2 normalization on q
        q_norm = F.normalize(q, p=self.options["p_norm"], dim=1)
        # L2 normalization on z
        z_norm = F.normalize(z, p=self.options["p_norm"], dim=1)
        # Return predictions (q) and target (z). Detach z from the graph since we won't update target using gradients
        return q_norm, z_norm.detach().clone()


class EncodeProject(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: tensor projection: (batch size, feature dim)
    """
    def __init__(self, options):
        super(EncodeProject, self).__init__()
        # If resnet==True, use Resnet as backbone. Else, use custom Encoder
        self.encoder = get_resnet(options) if options["resnet"] else CNNEncoder(options)
        # Project representation
        self.projector = Projector(options)

    def forward(self, x):
        # Forward pass on Encoder
        representation = self.encoder(x)
        # Compute latent by sampling if the model is VAE, or else the latent is just the mean.
        projection = self.projector(representation)
        # Return
        return projection


class Projector(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: tensor projection: (batch size, feature dim)
    """
    def __init__(self, options):
        super(Projector, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)
        # Forward pass on hidden layers
        self.hidden_layers = HiddenLayers(self.options)
        # Compute the mean i.e. bottleneck in Autoencoder
        self.projection = nn.Linear(self.options["dims"][-2], self.options["dims"][-1])

    def forward(self, h):
        # If defined, add gaussian noise at the input of encoder
        bs, _ = h.size()
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute the mean i.e. bottleneck in Autoencoder
        projection = self.projection(h)
        return projection


class Predictor(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: tensor predictions: (batch size, feature dim)
    """
    def __init__(self, options):
        super(Predictor, self).__init__()

        self.options = copy.deepcopy(options)
        projection_dim = self.options["dims"][-1]
        # Add hidden layers
        self.l1 = nn.Linear(projection_dim, 2*projection_dim)
        self.predictions = nn.Linear(2*projection_dim, projection_dim)

    def forward(self, h):
        h = F.relu(self.l1(h))
        predictions = self.predictions(h)
        return predictions


class CNNEncoder(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: Representation (h).

    Encoder is used for learning representations in Self-Supervised setting.
    It gets transformed images, and returns the representation of the data.
    """
    def __init__(self, options):
        super(CNNEncoder, self).__init__()
        # Container to hold layers of the architecture in order
        self.layers = nn.ModuleList()
        # Get configuration that contains architecture and hyper-parameters
        self.options = copy.deepcopy(options)
        # Get the dimensions of all layers
        dims = options["conv_dims"]
        # Input image size. Example: 28 for a 28x28 image.
        img_size = self.options["img_size"]
        # Get dimensions for convolution layers in the following format: [i, o, k, s, p, d]
        # i=input channel, o=output channel, k = kernel size, s = stride, p = padding, d = dilation
        convolution_layers = dims[:-1]
        # Final output dimension of encoder i.e. dimension of projection head
        output_dim = dims[-1]
        # Go through convolutional layers
        for layer_dims in convolution_layers:
            i, o, k, s, p, d = layer_dims
            self.layers.append(nn.Conv2d(i, o, k, stride=s, padding=p, dilation=d))
            # BatchNorm if True
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm2d(o))
            # Add activation
            self.layers.append(nn.LeakyReLU(inplace=False))
            # Dropout if True
            if options["isDropout"]:
                self.layers.append(nn.Dropout2d(options["dropout_rate"]))
        # Do global average pooling over spatial dimensions to make Encoder agnostic to input image size
        self.global_ave_pool = global_ave_pool

    def forward(self, x):
        # batch size, height, width, channel of the input
        bs, h, w, ch = x.size()
        # Forward pass on convolutional layers
        for layer in self.layers:
            x = layer(x)
        # Global average pooling over spatial dimensions. This is also used as learned representation.
        h = self.global_ave_pool(x)
        return h


class HiddenLayers(nn.Module):
    """
    :param dict options: Generic dictionary to configure the model for training.
    :return: tensor x: (batch size, feature dim)

    Used to build internal hidden layers of other MLP-based networks. Dimensions of each layer are defined
    as a list, dims, within ./config/byol.yaml
    """
    def __init__(self, options):
        super(HiddenLayers, self).__init__()
        self.layers = nn.ModuleList()
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

        dims = options["dims"]

        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm1d(dims[i]))

            self.layers.append(nn.LeakyReLU(inplace=False))
            if options["isDropout"]:
                self.layers.append(nn.Dropout(options["dropout_rate"]))

    def forward(self, x):
        for layer in self.layers:
            # You could do an if isinstance(layer, nn.Type1) maybe to check for types
            x = layer(x)
        return x


def get_resnet(config):
    # Get name of the resnet backbone
    net_name = config['backbone']
    # Initialize backbone
    backbone = models.__dict__[net_name](pretrained=config['pretrain'])
    # Build the resnet model from list of layers after removing last layer
    resnet = nn.Sequential(*list(backbone.children())[:-1])
    return resnet


class Flatten(nn.Module):
    "Flattens tensor to 2D: (batch_size, feature dim)"
    def forward(self, x):
        return x.view(x.shape[0], -1)

def global_ave_pool(x):
    """Global Average pooling of convolutional layers over the spatioal dimensions.
    Results in 2D tensor with dimension: (batch_size, number of channels) """
    return th.mean(x, dim=[2, 3])
