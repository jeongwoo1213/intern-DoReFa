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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# train setting
def train():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = trainloader(args)
    test_loader = testloader(args)

    # r_model = ResNet20(args)
    q_model = QResNet20(args)

    # r_model.to(device)
    q_model.to(device)

    # r_criterion = nn.CrossEntropyLoss()
    q_criterion = nn.CrossEntropyLoss()

    # r_optimizer = optim.Adam(params=r_model.parameters(), lr=args.lr)
    q_optimizer = optim.Adam(params=q_model.parameters(), lr=args.lr)



    # train
    for epoch in range(1,args.epochs+1):
        
        # r_model.train()
        q_model.train()

        for images,labels in train_loader:
            
            images = images.to(device)
            labels = labels.to(device)            
            
            # r_pred = r_model(images)
            q_pred = q_model(images)

            # r_loss = r_criterion(r_pred,labels)
            q_loss = q_criterion(q_pred,labels)
            
            # r_optimizer.zero_grad()
            q_optimizer.zero_grad()

            # r_loss.backward()
            q_loss.backward()
            
            # r_optimizer.step()
            q_optimizer.step()


        # print(f'epoch {epoch:<3}    fp loss {r_loss.item():<25}  q loss {q_loss.item()}')
        print(f'epoch {epoch:<3}    q loss {q_loss.item()}')
        
        if epoch % 10 == 0:
            
            # r_model.eval()
            q_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in testloader:
                    outputs = q_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of quantized network: {100 * correct / total} %')

if __name__=='__main__':
    train()

