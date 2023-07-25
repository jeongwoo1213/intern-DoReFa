import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ResNet20, QResNet20
from dataset import *


parser = argparse.ArgumentParser(description="PyTorch Implementation of DoReFa-Net")

# argument for train settings
parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers size')


# argument for quantization
parser.add_argument('--weight_bits', type=int, default=1, help='weights quantization bits')
parser.add_argument('--act_bits', type=int, default=2, help='activations quantization bits')

args = parser.parse_args()


if args.dataset == 'cifar10':
    args.num_classes = 10
elif args.dataset == 'cifar100':
    args.num_classes = 100

# train setting
def train():
    

    train_loader = trainloader(args)
    test_loader = testloader(args)

    r_model = ResNet20(args)
    q_model = QResNet20(args)

    r_criterion = nn.CrossEntropyLoss()
    q_criterion = nn.CrossEntropyLoss()

    r_optimizer = optim.Adam(params=r_model.parameters(), lr=args.lr)
    q_optimizer = optim.Adam(params=q_model.parameters(), lr=args.lr)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    r_model.to(device)
    q_model.to(device)


    # train
    for epoch in range(1,args.epochs+1):
        for images,labels in train_loader:
            
            images.to(device)
            labels.to(device)            
            
            r_pred = r_model(images)
            q_pred = q_model(images)

            r_loss = r_criterion(r_pred,labels)
            q_loss = q_criterion(q_pred,labels)
            
            r_optimizer.zero_grad()
            q_optimizer.zero_grad()

            r_loss.backward()
            q_loss.backward()
            
            r_optimizer.step()
            q_optimizer.step()


        print(f'epoch {epoch:<3}    fp loss {r_loss.item():<25}  q loss {q_loss.item()}')
           

if __name__=='__main__':
    train()

