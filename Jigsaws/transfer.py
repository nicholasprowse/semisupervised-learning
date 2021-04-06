#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program takes a network that was trained using the pretext task of solving jigsaws, and uses transfer learning
to train it to classify the STL10 images, using the 500 labelled training images
"""

import torch
from torchvision.datasets import STL10 as STL10
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import matplotlib
import argparse
import torch.optim as optim
import time
import os


# Trains the network for one epoch and returns the loss and accuracy on the training data
def train(net, loader, optimiser, loss_fun, device):
    net.train()
    acc = 0
    loss = 0
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        y_hat = net(x)
        batch_loss = loss_fun(y_hat, y)
        optimiser.zero_grad()
        batch_loss.backward()
        optimiser.step()
        prediction = torch.argmax(y_hat, 1)
        acc += sum((prediction == y).float()).item() / len(y)
        loss += batch_loss.item() / len(y)
    return loss / len(loader), acc / len(loader)


# Computes the loss and classification accuracy of the network on a given data loader
def get_accuracy(net, loader, loss_fun, device):
    net.eval()
    accuracy = 0
    loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            prediction = torch.argmax(y_hat, 1)
            loss += loss_fun(y_hat, y.long()).item() / len(y)
            accuracy += sum((prediction == y).float()).item() / len(y)
    return loss / len(loader), accuracy / len(loader)


# Consists of a single ResNet with the fc layer removed, and a 4608x100 fc layer. Takes nine images as input, each is
# passed through the ResNet separately, then each of the 512 dimensional outputs are concatenated and passed through the
# fc layer
class CombinedResNet(nn.Module):
    def __init__(self):
        super(CombinedResNet, self).__init__()
        full_resnet = torchvision.models.resnet18()
        self.resnet = nn.Sequential(*list(full_resnet.children())[0:-1])
        num_features = full_resnet.fc.in_features
        self.fc = nn.Linear(9 * num_features, 100)

    def forward(self, x):
        x1 = torch.zeros(x.shape[0], 9, 512, 1, 1).to(x.device)
        for i in range(9):
            x1[:, i] = self.resnet(x[:, i])
        x = x1.reshape(x1.shape[0], -1)
        return self.fc(x)


# This is a model which takes an existing network, and adds a fc layer to the end of it. 
# Used to create the ResNet18 from the network loaded from the pretext task
class TransferNet(nn.Module):
    def __init__(self, net):
        super(TransferNet, self).__init__()
        self.resnet = net
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


def main():
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Train segmentation model.")

    parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("--epochs", help="num training epochs", type=int, default=100)
    # lr is learning rate used to train last layer, and fine_lr is learning rate used to fine tune entire network
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-2)
    parser.add_argument("--fine_lr", help="learning rate", type=float, default=5e-4)

    args = parser.parse_args()

    # For MonARCH
    dataset_dir = "/mnt/lustre/projects/ds19/SHARED"

    # All images are 3x96x96
    image_size = 96
    batch_size = args.batch_size
    lr = args.lr

    gpu_idx = 0
    device = torch.device(gpu_idx if torch.cuda.is_available() else 'cpu')
    print(device)

    # Perform random crops and mirroring for data augmentation
    transform_train = transforms.Compose(
        [transforms.RandomCrop(image_size, padding=4),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # No random
    transform_test = transforms.Compose(
        [transforms.CenterCrop(image_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    start_path = 'jigsaw.pt'
    save_path = 'classifier.pt'

    # Existing network is a CombinedResNet
    net = CombinedResNet().to(device)

    # Load existing network (the one trained on the pretext task), and convert the ResNet contained in it into a
    # TransferNet to add the fc layer back onto it
    if os.path.isfile(start_path):
        # Raise Error if it does not exist
        print("Checkpoint does not exist, starting from scratch")
        check_point = torch.load(start_path)
        net.load_state_dict(check_point['model_state_dict'])
        net = TransferNet(net.resnet).to(device)
        net.resnet.requires_grad = False
    else:
        raise ValueError("Rotation classifier does not exist")

    optimiser = optim.Adam(net.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()

    # Create training and validation split
    # Load train and validation sets
    train_val_set = STL10(dataset_dir, split='train', transform=transform_train, download=False)

    # Use 10% of data for training - simulating low data scenario
    num_train = int(len(train_val_set) * 0.1)

    # Split data into train/val sets
    # Set torch's random seed so that random split of data is reproducible
    torch.manual_seed(0)
    train_set, val_set = random_split(train_val_set, [num_train, len(train_val_set) - num_train])

    test_set = STL10(dataset_dir, split='test', transform=transform_test, download=False)

    # Create the 3 data loaders
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    epochs = args.epochs
    train_loss_logger = [0] * 2 * epochs
    val_loss_logger = [0] * 2 * epochs
    train_acc_logger = [0] * 2 * epochs
    val_acc_logger = [0] * 2 * epochs
    test_acc_logger = [0] * 2 * epochs

    # Training loop
    for epoch in range(2 * epochs):
        start = time.time()
        train_loss_logger[epoch], train_acc_logger[epoch] = train(net, train_loader, optimiser, loss_fun, device)
        val_loss_logger[epoch], val_acc_logger[epoch] = get_accuracy(net, valid_loader, loss_fun, device)
        _, test_acc_logger[epoch] = get_accuracy(net, test_loader, loss_fun, device)

        # Unfreeze network after 100 epochs, and set the learning rate to the fine tuning learning rate
        if epoch % 100 == 99:
            lr = args.fine_lr
            net.resnet.requires_grad = True
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

        epoch_time = time.time() - start
        print(
            'Epoch: {}, Loss: {:4.3e}, Val Loss: {:4.3e}, Training accuracy: {:4.3f},'
            'Validation accuracy: {:4.3f}, Time: {:4.3f}s'
            .format(epoch, train_loss_logger[epoch], val_loss_logger[epoch], train_acc_logger[epoch],
                    val_acc_logger[epoch], epoch_time))

        # Save checkpoint
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'train_acc': train_acc_logger,
            'train_loss': train_loss_logger,
            'valid_acc': val_acc_logger,
            'valid_loss': val_loss_logger,
            'test_acc': test_acc_logger
        }, save_path)

    # Plot results
    print(f"Test accuracy: {test_acc_logger[-1]}")
    plt.figure()
    plt.plot(train_loss_logger)
    plt.plot(val_loss_logger)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Classification loss with jigsaws")
    plt.savefig("class_jigsaw_loss.pdf")

    plt.figure()
    plt.plot(train_acc_logger)
    plt.plot(val_acc_logger)
    plt.plot(test_acc_logger)
    plt.legend(["Training Accuracy", "Validation Accuracy", "Test Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Classification accuracy with jigsaws")
    plt.savefig("class_jigsaw_acc.pdf")


if __name__ == '__main__':
    main()
