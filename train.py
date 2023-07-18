import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import Resnet20, QResnet20


# train setting
def train():
    epochs = 100
    lr = 1e-3

    batch_size = 16
    num_classes = 10

    x = torch.rand(batch_size,3,32,32)

    r_model = Resnet20()
    q_model = QResnet20()

    r_criterion = nn.CrossEntropyLoss()
    q_criterion = nn.CrossEntropyLoss()

    r_optimizer = optim.Adam(params=r_model.parameters(), lr=lr)
    q_optimizer = optim.Adam(params=q_model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    r_model.to(device)
    q_model.to(device)

    label = torch.floor(torch.rand(batch_size)*num_classes).long()

    for epoch in range(1,epochs+1):
        x = x.to(device)
        # x_prime = x.clone().to(device)
        label = label.to(device)
        # label_prime = label.clone().to(device)
        
        
        r_pred = r_model(x)
        q_pred = q_model(x)

        r_loss = r_criterion(r_pred,label)
        q_loss = q_criterion(q_pred,label)
        
        r_optimizer.zero_grad()
        q_optimizer.zero_grad()

        r_loss.backward()
        q_loss.backward()
        
        r_optimizer.step()
        q_optimizer.step()

        if (epoch) % 10 == 0:
           print(f'epoch {epoch}   r loss {r_loss.item()}    q loss {q_loss.item()}')
           






if __name__=="__main__":

    train()
