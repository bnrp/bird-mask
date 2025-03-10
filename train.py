import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random

torch.cuda.empty_cache()

from bird_mask_net import BasicBlock, ResNet, BetterBlock, ResNetTransfer
from train_utils import train, validate
from utils import load_data, load_train_test
from nabirdsDataset import nabirdsDataset
from simple_conv_net import MyModel
from resnet import resnet20 

dataset_path = 'nabirds-data/nabirds/'
image_path = dataset_path + 'images/'

seed = 848188
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

validate_mod = 99

epochs = 100
batch_size = 32
learning_rate = 0.05
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_images, test_images = load_train_test(dataset_path)
#print(len(train_images))
#print(len(test_images))

train_data = nabirdsDataset(dataset_path, image_path, ignore=test_images, general=False)
test_data = nabirdsDataset(dataset_path, image_path, ignore=train_images, general=False)

classes = train_data.image_class_labels_rectified['rectified_id']
unique = classes.value_counts()
unique = unique.sort_index()
classes = (unique.index.values[-1] + 1) 
weights = unique.to_numpy()
weights = weights / np.sum(weights)
weights = weights * classes 
weights = torch.tensor(weights).to(device)
#print(weights)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)


#model = ResNet(img_channels=3, num_layers=50, block=BetterBlock, num_classes=555)
model = resnet20(num_classes=555)
#container = ResNetTransfer(backbone="resnet50", load_pretrained=True, classes=555) 
#container.fine_tune(what="NEW_LAYERS")
#model = container.model
model.to(device)


print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} total trainable parameters.")

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(model.parameters(), 0.01, amsgrad=True, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss(weight=weights)

#print(torch.cuda.memory_summary())



if __name__ == "__main__":
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)

        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        if (epoch + 1) % validate_mod == 0:
            valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device)
            valid_loss.append(valid_epoch_loss)
            valid_acc.append(valid_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        if (epoch + 1) % validate_mod == 0:
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        print('TRAINING COMPLETE')
