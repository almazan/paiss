# archs.py
import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import models
from modules import GeneralizedMeanPooling
import os.path as osp
import sys

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_RMAC(nn.Module):

    def __init__(self, block, layers, fc_out, norm_features=True, aggregation='gem',
                 dropout_p=None, gemp=3, without_fc=False):
        self.inplanes = 64
        self.norm_features = norm_features
        self.aggregation = aggregation
        self.without_fc = without_fc
        self.out_features = fc_out
        super(ResNet_RMAC, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Aggregation layer
        if aggregation == None:
            self.adpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        elif aggregation == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif aggregation == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif aggregation == 'gem':
            self.adpool = GeneralizedMeanPooling(norm_type=gemp, output_size=1)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

        # Final FC layer
        self.fc = nn.Linear(512 * block.expansion, fc_out) if not self.without_fc else None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        # Extract convolutional features
        x = self.features(x)
        # Aggregate features into a 1D representation
        x = self.adpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x.squeeze_()
        # Projection (if defined)
        if not self.without_fc:
            x = self.fc(x)
        # L2-normalize the representation
        x = l2_normalize(x)

        return x

def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x

def resnet101_rmac(out_dim=2048, dropout_p=None, weights=None, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet_RMAC(Bottleneck, [3, 4, 23, 3], dropout_p=dropout_p, fc_out=out_dim, **kwargs)
    if weights:
        try:
            if torch.cuda.device_count()>0:
                weight_dict = torch.load(weights)['state_dict']
            else:
                weight_dict = torch.load(weights, map_location={'cuda:0':'cpu'})['state_dict']
        except OSError as e:
            print ('ERROR: Weights {} not found. Please follow the instructions to download models.'.format(weights))
            sys.exit()
        model.load_state_dict(weight_dict)
    return model

def resnet50_rmac(out_dim=2048, dropout_p=None, weights=None, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet_RMAC(Bottleneck, [3, 4, 6, 3], dropout_p=dropout_p, fc_out=out_dim, **kwargs)
    if weights:
        try:
            if torch.cuda.device_count()>0:
                weight_dict = torch.load(weights)['state_dict']
            else:
                weight_dict = torch.load(weights, map_location={'cuda:0':'cpu'})['state_dict']
        except OSError as e:
            print ('ERROR: Weights {} not found. Please follow the instructions to download models.'.format(weights))
            sys.exit()
        model.load_state_dict(weight_dict)
    return model

def resnet18_rmac(out_dim=512, dropout_p=None, weights=None, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet_RMAC(BasicBlock, [2, 2, 2, 2], dropout_p=dropout_p, fc_out=out_dim, **kwargs)
    if weights:
        try:
            if torch.cuda.device_count()>0:
                weight_dict = torch.load(weights)['state_dict']
            else:
                weight_dict = torch.load(weights, map_location={'cuda:0':'cpu'})['state_dict']
        except OSError as e:
            print ('ERROR: Weights {} not found. Please follow the instructions to download models.'.format(weights))
            sys.exit()
        model.load_state_dict(weight_dict)
    return model

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, training=True):
        super(AlexNet, self).__init__()
        self.training = training
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if self.training:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if not self.training:
            x = l2_normalize(x)
        return x

class AlexNet_RMAC(nn.Module):

    def __init__(self, aggregation='gem', gemp=3, norm_features=True, showforward=False):
        super(AlexNet_RMAC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        if aggregation == 'gem':
            self.features.add_module('12', GeneralizedMeanPooling(norm_type=gemp, output_size=1))
        else:
            self.features.add_module('12', nn.AdaptiveAvgPool2d(output_size=1))
        self.norm_features = norm_features
        self.showforward = showforward

    def forward(self, x):
        if self.showforward: print('Input size of model.features : '+str(x.shape)+'\n')
        x = self.features(x)
        x = x.view(x.shape[0], x.shape[1])
        if self.norm_features:
            x = l2_normalize(x)
        if self.showforward: print('Output size of model.features : '+str(x.shape)+'\n')
        return x

def alexnet_fc(out_dim=1000, train=True, weights=None, **kwargs):
    model = AlexNet(num_classes=out_dim, training=train)
    model.out_features = out_dim
    if weights:
        if torch.cuda.device_count()>0:
            weight_dict = torch.load(weights)
        else:
            weight_dict = torch.load(weights, map_location={'cuda:0':'cpu'})
        if not train:
            del weight_dict['classifier.6.weight']
            del weight_dict['classifier.6.bias']
        model.load_state_dict(weight_dict)
    return model

def alexnet_rmac(aggregation='gem',weights=None, **kwargs):
    model = AlexNet_RMAC(aggregation=aggregation)
    model.out_features = 256
    if weights:
        if torch.cuda.device_count()>0:
            weight_dict = torch.load(weights)
        else:
            weight_dict = torch.load(weights, map_location={'cuda:0':'cpu'})
        del weight_dict['classifier.1.weight']; del weight_dict['classifier.1.bias']
        del weight_dict['classifier.4.weight']; del weight_dict['classifier.4.bias']
        del weight_dict['classifier.6.weight']; del weight_dict['classifier.6.bias']
        model.load_state_dict(weight_dict)
    return model

# models to be called
path_models = 'data/models'

resnet50_cls = lambda : resnet50_rmac(out_dim=586, weights=osp.join(path_models, 'resnet50-cls-lm.pt'))
resnet18_rank_DA = lambda : resnet18_rmac(out_dim=512, weights=osp.join(path_models, 'resnet18-rnk-lm-da.pt'))
resnet50_rank = lambda : resnet50_rmac(out_dim=2048, weights=osp.join(path_models, 'resnet50-rnk-lm.pt'))
resnet50_rank_DA = lambda : resnet50_rmac(out_dim=2048, weights=osp.join(path_models, 'resnet50-rnk-lm-da.pt'))

# weightless architectures to be called
alexnet_imagenet = lambda: alexnet_fc(out_dim=1000, train=True)
alexnet_imagenet_fc7 = lambda: alexnet_fc(out_dim=4096, train=False)
alexnet_lm = lambda: alexnet_fc(out_dim=586, train=True)
alexnet_GeM = lambda: alexnet_rmac()
resnet18 =  lambda : resnet18_rmac(out_dim=586, aggregation=None)
resnet18_GeM = lambda : resnet18_rmac(out_dim=512)
