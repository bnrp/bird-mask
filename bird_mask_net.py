import nabirds
from nabirdsDataset import nabirdsDataset
import utils
import train_utils

import time

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils
from torch.utils.data import DataLoader
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2

if torch.cuda.is_available():
    device=torch.device('cuda')
    print('CUDA enabled')
else:
    device=torch.device('cpu')
    print('CUDA not available')

# https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
# https://d2l.ai/chapter_convolutional-modern/resnet.html
# https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.expansion = expansion
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out



class BetterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, downsample=None, first=False):
        super(BetterBlock, self).__init__()

        self.expansion = expansion
        self.downsample = downsample
        self.out_channels = out_channels
       
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion) 

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out




class ResNet(nn.Module):
    def __init__(self, img_channels, num_layers, block, num_classes=1000):
        super(ResNet, self).__init__()
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1
        elif num_layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4

        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*self.expansion, num_classes)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels*self.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels*self.expansion))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print('Dimensions of the first convolutional feature map: ', x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        #print('Dimensions of output: ', x.shape)

        return x


class ResNetTransfer():
    def __init__(self, backbone="resnet50", load_pretrained=True, classes=555):
        super().__init__()
        self.backbone = backbone
        self.pretrained_model = None
        self.classifier_layers = []
        self.new_layers = []
        self.classes = classes
        self.model = None

        if backbone == "resnet50":
            if load_pretrained:
                self.pretrained_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.pretrained_model = torchvision.models.resnet50(weights=None)

            self.classifier_layers = [self.pretrained_model.fc]
            self.pretrained_model.fc = nn.Linear(in_features=2048, out_features=classes, bias=True)
            self.new_layers = [self.pretrained_model.fc]
            self.model = self.pretrained_model

    def fine_tune(self, what=""):
        m = self.pretrained_model
        for p in m.parameters():
            p.requires_grad = False
        
        if what == "NEW_LAYERS":
            for l in self.new_layers:
                for p in l.parameters():
                    p.requires_grad = True
        elif what == "CLASSIFIER":
            for l in self.classifier_layers:
                for p in l.parameters():
                    p.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = True

    def get_optimizer_params(self):
        options = []
        if self.backbone in ["resnet50", "resnet152"]:
            layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
            lr = 0.0001
            for layer_name in reversed(layers):
                options.append({"params": getattr(self.pretrained_model, layer_name).parameters(), "lr": lr})
                lr = lr / 3.0

        return options




def main():
    # Vars
    batch_size = 128
    lr = 1e-2
    epochs = 3


    # Dataset locations
    dataset_path = 'nabirds-data/nabirds/'
    image_path = dataset_path + 'images/'

    # Load training/testing data
    train_images, test_images = load_train_test(dataset_path)

    train_data = nabirdsDataset(dataset_path, image_path, ignore=test_images)
    test_data = nabirdsDataset(dataset_path, image_path, ignore=train_images)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create model
    model_18 = ResNet(num_classes=train_data.classes)
    model_18 = model_18.to(device)

    # Train
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model_18.parameters(), lr=lr)
    train(model_18, epochs, train_data, test_data, criterion, optimizer)

    #model_18.load_state_dict(torch.load('best.mdl'))
    #evaluate(model_18, train_loader)


if __name__ == '__main__':
    tensor = torch.rand([1,3,224,224])
    model = ResNet(img_channels=3, num_layers=50, block=BetterBlock, num_classes=555)
    print(model)

    output = model(tensor)
