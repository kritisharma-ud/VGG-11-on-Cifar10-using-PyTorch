import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import torch.distributed as dist
import torch.utils.data.distributed
from vgg_handler import VGGHandler



device = "cpu"
torch.set_num_threads(4)

batch_size = 256 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    for epoch in range(1
                       ):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_idx % 20 == 19:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
    print('Finished Training')
    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy

def main():
    # changed parameters to achieve the highest accuracy
    learning_rate =0.01
    weight_decay = 1e-4
    #weight_decay = 0
   # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                #std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    #normalize = transforms.Normalize(mean=[x/255.0 for x in [0.4914, 0.4822, 0.4465]],
            #std=[x/255.0 for x in [0.2023, 0.1994, 0.2010]])

    # changed the normalize in terms to increase the accuracy 
    normalize=transforms.Normalize( 
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    # transform_train=transforms.Compose([
    # transforms.Resize(size=(224, 224)),
    # transforms.ToTensor(),normalize,
    #         ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    shuffle=True,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay,nesterov=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', factor=0.5)

    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_acc = test_model(model, test_loader, training_criterion)
        scheduler.step(test_acc)  # Update the learning rate scheduler

    if test_acc is not None:
        print(f"Accuracy achieved: {test_acc:.2f}%")


if __name__ == "__main__":
    main()