import nabirds
from nabirdsDataset import nabirdsDataset
import time

import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader

from PIL import Image
#import cv2
import numpy as np
from matplotlib import pyplot as plt

#import tensorflow as tf
#from tf.keras import layers
#from tf.keras import Model

# Enable CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA Enabled')
else:
    print('CUDA not available')
    #quit()


# From https://learnopencv.com/fully-convolutional-image-classification-on-arbitrary-sized-image/
class FCResNet(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        # Start with standard resnet18 defined here 
        super().__init__(block = models.resnet.BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url( models.resnet.model_urls["resnet50"], progress=True)
            self.load_state_dict(state_dict)
 
        # Replace AdaptiveAvgPool2d with standard AvgPool2d 
        self.avgpool = nn.AvgPool2d((3, 3))
 
        # Convert the original fc layer to a convolutional layer.  
        self.last_conv = torch.nn.Conv2d( in_channels = self.fc.in_features, out_channels = num_classes, kernel_size = 1)
        self.last_conv.weight.data.copy_( self.fc.weight.data.view ( *self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)
 
    # Reimplementing forward pass. 
    def forward(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
 
        # Notice, there is no forward pass 
        # through the original fully connected layer. 
        # Instead, we forward pass through the last conv layer
        x = self.last_conv(x)
        return x


def load_data(dataset_path, image_path):
    ### Data ###
    # Image Data
    # image_paths        - dictionary - {image_id: image_path}
    # image_sizes        - dictionary - {image_id: [width, height]}
    # image_bboxes       - dictionary - {image_id: bbox}
    # image_parts        - dictionary - {image_id: [part_id] = [x, y]}
    # image_class_labels - dictionary - {imqage_id: class_id}
    #
    # Class Data
    # class_names        - dictionary - {class_id: class_name}
    # class_hierarchy    - dictionary - {child_id: parent_id}

    #
    # Parts Data
    # part_names         - dictionary - {part_id: part_name}
    # part_ids           - list - sorted list of index integers

    # Load image data
    image_paths = nabirds.load_image_paths(dataset_path, image_path)
    image_sizes = nabirds.load_image_sizes(dataset_path)
    image_bboxes = nabirds.load_bounding_box_annotations(dataset_path)
    image_parts = nabirds.load_part_annotations(dataset_path)
    image_class_labels = nabirds.load_image_labels(dataset_path)

    # Load in the class data
    class_names = nabirds.load_class_names(dataset_path)
    class_hierarchy = nabirds.load_hierarchy(dataset_path)

    # Load in the part data
    part_names = nabirds.load_part_names(dataset_path)
    part_ids = list(part_names.keys())
    part_ids.sort() 

    return image_paths, image_sizes, image_bboxes, image_parts, image_class_labels, class_names, class_hierarchy, part_names, part_ids


def load_train_test(dataset_path):
    ### Data ###
    # train_images - list - image_ids for training set
    # test_images  - list - image_ids for testing set

    # Load in the train / test split
    train_images, test_images = nabirds.load_train_test_split(dataset_path)

    return train_images, test_images


def timer(func):
    def timer_wrap(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()

        print(func.__name__, round(t1 - t0, 3))
        return result

    return timer_wrap()


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
            
            if x.dim() == 3:
                x = x[None, :, :, :]

            logits = model(x)
            pred, class_idx = torch.max(logits, dim=1)
            row_max, row_idx = torch.max(pred, dim=1)
            col_max, col_idx = torch.max(row_max, dim=1)
            pred = logits[0,:,row_idx[0, int(col_idx)], int(col_idx)]
            loss = criterion(pred, y)

            #a = list(model.parameters())[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sample += 1
            
            #b = list(copy.deepcopy(model))[0]

            #print(torch.equal(a.data, b.data))
                        

            if (sample % 1000) == 0:
                print('Finished with', sample, 'samples!')

        if epoch % 1 == 0:
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
            
        if x.dim() == 3:
            x = x[None, :, :, :]
        
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
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
 
    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
       
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    # Vars
    batch_size = 32
    lr = 5e-3
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
    model_18 = FCResNet(num_classes=train_data.classes)
    model_18 = model_18.to('cuda')

    print(model_18)

    # Train
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model_18.parameters(), lr=lr)
    train(model_18, epochs, train_data, test_data, criterion, optimizer)

    #model_18.load_state_dict(torch.load('best.mdl'))
    #evaluate(model_18, train_loader)


if __name__ == '__main__':
    main()
