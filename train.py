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
parser.add_argument('--epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers size')


# argument for quantization
parser.add_argument('--weight_bits', type=int, default=1, help='weights quantization bits')
parser.add_argument('--act_bits', type=int, default=2, help='activations quantization bits')

# misc
parser.add_argument('--gpu_id', type=str, default='2', help='target GPU to use')


args = parser.parse_args()


if args.dataset == 'cifar10':
    args.num_classes = 10
elif args.dataset == 'cifar100':
    args.num_classes = 100


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


log_dir = f'../results/dorefa/ResNet20_{args.dataset}/W{args.weight_bits}A{args.act_bits}'

if not os.path.exists(log_dir):
    os.makedirs(os.path.join(log_dir, 'checkpoint'))

# train
def train():
    
    train_loader = trainloader(args)
    test_loader = testloader(args)
        
    if (args.weight_bits == 32) and (args.act_bits == 32):
        model = ResNet20(args)
    else:
        model = QResNet20(args)
        
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_accuracy = 0
    
    # train
    start = time.time()
    
    print("============== START TRAINING ==============")
    
    for epoch in range(1, args.epochs+1):
        
        model.train()
        
        for images,labels in train_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            pred = model(images)
            
            loss = criterion(pred,labels)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            
        # validation
        
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _,predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 * correct / total
        best_accuracy = max(accuracy, best_accuracy)
                
        if accuracy == best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch':epoch,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'criterion':criterion.state_dict()
            }, os.path.join(log_dir,'checkpoint/best_checkpoint.pth'))
            
            
        if epoch % 5 == 0: 
            elapsed_time = time.ctime(time.time() - start).split(' ')[4]
            print(f'Epoch {epoch:<3}    loss {loss.item()}  time {elapsed_time}')
        
        if epoch % 20 == 0:
            print(f'Accuracy of network: {accuracy:.2f} %\n')
            
            
            
    print(f'best accuracy of W{args.weight_bits}A{args.act_bits}: {best_accuracy:.2f}')


if __name__=='__main__':
    train()
