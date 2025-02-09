import nabirds
from nabirdsDataset import nabirdsDataset
from utils import load_train_test, timer, load_data
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
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if downsample is not None:
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

        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 51, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*self.expansion, num_classes)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels*self.expansion))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channels, expansion=self.expanion))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



def train(model, epochs, train_loader, test_loader, criterion, optimizer):
    best_accuracy = 0
    best_epoch = 0

    for epoch in range(epochs):
        print('Starting Epoch', epoch)
        t0 = time.time()
        sample = 0
        for step, (x,y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            pred, class_idx = torch.max(logits, dim=1)
            row_max, row_idx = torch.max(pred, dim=1)
            col_max, col_idx = torch.max(row_max, dim=1)
            pred = logits[0,:,row_idx[0, col_idx], col_idx]
            loss = criterion(pred, y)

            a = list(model.parameters())[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sample += 1
            
            b = list(model.parameters(),clone())[0]

            #print(torch.equal(a.data, b.data))
                        

            if (sample % 1000) == 0:
                print('Finished with', sample, 'samples!')

        if epoch % 1 == 0:
            model.eval()
            accuracy = evaluate(model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch

                torch.save(model.state_dict(), 'best.mdl')
        print(round(time.time()-t0, 3))
        print('Completed Epoch', epoch)
    print('best acc:', best_accuracy, 'best_epoch:', best_epoch)


def evaluate(model, loader):
    model.eval()

    correct = 0
    sample = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred, class_idx = torch.max(logits, dim=1)
            row_max, row_idx = torch.max(pred, dim=1)
            col_max, col_idx = torch.max(row_max, dim=1)
            pred = logits[0,:,row_idx[0, col_idx], col_idx]
            pred = torch.argmax(pred)

        correct += torch.eq(pred,torch.argmax(y))

        sample += 1

        if (sample % 1000) == 0:
                print('Evaluated', sample, 'samples!', 'Current acc:', correct / sample)

    return correct/sample


def main():
    # Vars
    batch_size = 128
    lr = 1e-3
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
    main()
