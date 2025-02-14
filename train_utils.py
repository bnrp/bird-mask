import torch
import torch.optim as optim

from tqdm import tqdm


def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training...')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.00001, T_max=100*len(trainloader))

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1

        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(image)
        #print(outputs)
        #print(outputs.shape)
        #print(labels)
        #print(labels.shape)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        train_running_correct += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()
        sched.step()

    epoch_loss = train_running_loss / counter
    print('-'*50)
    print('Correct Train:')
    print(train_running_correct)
    print('Total Train:')
    print(len(trainloader.dataset))
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc


def validate(model, testloader, criterion, device):
    model.eval()
    print('Validating...')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    return epoch_loss, epoch_acc
        
