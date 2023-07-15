import os
import time
import argparse


from model import *


# train setting
def train():
    epochs = 100
    lr = 1e-3

    batch_size = 128
    num_classes = 10

    x = torch.rand(batch_size,1,28,28)

    r_model = R_Test(num_classes=num_classes)
    q_model = Q_Test(num_classes=num_classes)

    r_criterion = nn.CrossEntropyLoss()
    q_criterion = nn.CrossEntropyLoss()

    r_optimizer = optim.Adam(params=r_model.parameters(), lr=lr)
    q_optimizer = optim.Adam(params=q_model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    r_model.to(device)
    q_model.to(device)

    label = torch.floor(torch.rand(batch_size)*num_classes).long()

    for epoch in range(epochs):
        x = x.to(device)
        label = label.to(device)
        
        
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

        if epoch % 10 == 0:
           print(f'epoch {epoch}   r loss {r_loss.item()}    q loss {q_loss.item()}')
           






if __name__=="__main__":
    # x = torch.rand(1,1,28,28)

    # # print(torch.mean(x))

    # model = Test(num_classes=5)



    # print(model(x))

    # x = np.array([[0.9296, 0.8788, 0.4283],
    #       [0.2223, 0.5305, 0.3603],
    #       [0.3625, 0.1576, 0.0012]])
    
    # z = np.array([[-0.2056, -0.2056,  0.2056],
    #       [ 0.2056,  0.2056, -0.2056],
    #       [ 0.2056,  0.2056,  0.2056]])
    
    
    # # print(x*z)
    # # print(np.sum(x*z))

    train()
