#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains a model to perform the pretext task of solving jigsaws of the images

We will be using a dataset that can be obtained directly from the torchvision package. There are 10 classes and we will
be training a CNN for the image classification task. We have training, validation and test sets that are labelled with
the class, and a large unlabeled set.

We will simulating a low training data scenario by only sampling a small percentage of the labelled data (10%) as
training data. The remaining examples will be used as the validation set.

To get the labelled data, change the dataset_dir to something suitable for your machine, and execute the following
(you will then probably want to wrap the dataset objects in a PyTorch DataLoader):
"""

import torch
from torchvision.datasets import STL10 as STL10
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torchvision
import torch.optim as optim
import random
import time
import argparse
import os
import sys

# For MonARCH
dataset_dir = "/mnt/lustre/projects/ds19/SHARED"

# All images are 3x96x96
image_size = 96


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


# Transformation that randomly converts an image to grayscale with a probability of 2/3
class DropColor:
    def __init__(self):
        self.gray = transforms.Grayscale(num_output_channels=3)

    def __call__(self, x):
        if random.random() > 2 / 3:
            return x
        return self.gray(x)


# Dataset that returns data points of size 9x3x96x96. That is, 9 images, each of which is a small patch of a larger 
# image. The 9 images are jumbled up according to one of a 100 predefined permutations, and the index of the applied 
# permutation is provided as the label
class JigsawDataset(STL10):
    # pixel locations for each of the patches
    top = [0, 0, 0, 30, 30, 30, 60, 60, 60]
    left = [0, 30, 60, 0, 30, 60, 0, 30, 60]
    perms = np.load('perms.npy')
    
    # If rand=False, then the permutation applied is always the same (i.e. for testing/validation)
    # If rand=True, then a random permutation (from the perms set) is applied (i.e. for training)
    # Pretransforms are done before the image is split up
    # Post transforms are done after the image is split up, and are performed on each image separatly
    def __init__(self, split, rand, pre_transform, post_transform):
        super().__init__(dataset_dir, split=split, transform=pre_transform, download=False)
        self.post_transform = post_transform
        self.random = rand

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if self.random:
            perm_idx = random.randint(0, 99)
        else:
            perm_idx = idx % 99
        perm = JigsawDataset.perms[perm_idx]
        patches = torch.zeros([9, 3, image_size, image_size])
        for i in range(9):
            patch = functional.crop(image, JigsawDataset.top[perm[i]], JigsawDataset.left[perm[i]], 30, 30)
            patches[i] = self.post_transform(patch).unsqueeze(0)
        return patches, perm_idx


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


def main():
    # Save start time. Program stops after 57 minutes so that we don't lose the output on MonArch
    program_time = time.time()

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Train segmentation model.")
    
    parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("--epochs", help="num training epochs", type=int, default=30)
    parser.add_argument("--lr", help="learning rate", type=float, default=5e-3)
    parser.add_argument("--load_checkpoint", action='store_true', help="Load checkpoint or start from scratch")

    args = parser.parse_args()
    
    batch_size = args.batch_size
    lr = args.lr
    
    gpu_idx = 0
    device = torch.device(gpu_idx if torch.cuda.is_available() else 'cpu')
    print(device)

    # Transforms for the datasets
    pre_transform_random = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(90)])

    post_transform_random = transforms.Compose([
        transforms.RandomCrop(26),
        transforms.Resize(image_size),
        DropColor(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    pre_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(90)])

    post_transform = transforms.Compose([
        transforms.CenterCrop(26),
        transforms.Resize(image_size),
        DropColor(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create training and validation split
    # Load train and validation sets
    train_val_set = JigsawDataset('train', False, pre_transform, post_transform)

    # Use 10% of data for training - simulating low data scenario
    num_train = int(len(train_val_set) * 0.1)

    # Split data into train/val sets
    # Set torch's random seed so that random split of data is reproducible
    torch.manual_seed(0)
    _, val_set = random_split(train_val_set, [num_train, len(train_val_set) - num_train])
    
    unlabelled_set = JigsawDataset('unlabeled', True, pre_transform_random, post_transform_random)

    # Create the 2 data loaders
    unlabelled_loader = DataLoader(unlabelled_set, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(val_set, batch_size=batch_size)
    
    net = CombinedResNet().to(device)
    epochs = args.epochs

    train_loss_logger = [0.] * epochs
    train_acc_logger = [0.] * epochs
    val_loss_logger = [0.] * epochs
    val_acc_logger = [0.] * epochs

    optimiser = optim.Adam(net.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()

    start_epoch = 0
    save_path = 'jigsaw.pt'

    # Load Checkpoint
    if args.load_checkpoint:
        # Check if checkpoint exists
        if os.path.isfile(save_path):
            # load Checkpoint
            check_point = torch.load(save_path)
            # Checkpoint is saved as a python dictionary
            # https://www.w3schools.com/python/python_dictionaries.asp
            # here we unpack the dictionary to get our previous training states
            net.load_state_dict(check_point['model_state_dict'])
            optimiser.load_state_dict(check_point['optimiser_state_dict'])
            start_epoch = check_point['epoch']
            train_loss_logger = check_point['train_loss']
            train_acc_logger = check_point['train_acc']
            val_loss_logger = check_point['valid_loss']
            val_acc_logger = check_point['valid_acc']
            lr = check_point['lr']

            print("Checkpoint loaded, starting from epoch:", start_epoch)
        else:
            # Raise Error if it does not exist
            print("Checkpoint does not exist, starting from scratch")
    else:
        # If checkpoint does exist and Start_From_Checkpoint = False
        # Raise an error to prevent accidental overwriting
        if os.path.isfile(save_path):
            raise ValueError("Warning Checkpoint exists")
        else:
            print("Starting from scratch")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        start = time.time()
        train_loss_logger[epoch], train_acc_logger[epoch] = train(net, unlabelled_loader, optimiser, loss_fun, device)
        val_loss_logger[epoch], val_acc_logger[epoch] = get_accuracy(net, valid_loader, loss_fun, device)

        # Reduce learning rate every 10 epochs
        if epoch % 10 == 9:
            lr *= 0.5
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr

        epoch_time = time.time() - start
        print('Epoch: {}, Training Loss: {:4.3e}, Validation Loss: {:4.3e}, Training Accuracy: {:4.3f}, '
              'Validation Accuracy: {:4.3f}, Time: {:4.3f}s'
              .format(epoch, train_loss_logger[epoch], val_loss_logger[epoch], train_acc_logger[epoch],
                      val_acc_logger[epoch], epoch_time))

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'train_acc': train_acc_logger,
            'train_loss': train_loss_logger,
            'valid_acc': val_acc_logger,
            'valid_loss': val_loss_logger,
            'lr': lr
        }, save_path)

        # If we cannot complete the next epoch before 57 minutes, quit since MonArch will cut us off after 1 hour
        if (time.time() - program_time) + epoch_time > 60 * 57:
            sys.exit(0)

    # Plot loss and accuracy
    plt.figure()
    plt.plot(train_loss_logger)
    plt.plot(val_loss_logger)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss of network solving Jigsaw")
    plt.savefig("jigsaw_loss.pdf")

    plt.figure()
    plt.plot(train_acc_logger)
    plt.plot(val_acc_logger)
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy of network solving Jigsaw")
    plt.savefig("jigsaw_acc.pdf")


if __name__ == '__main__':
    main()
