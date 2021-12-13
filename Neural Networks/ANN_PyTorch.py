
## Machine Learning HW 5 - ANN implementation using PyTorch
## Author: Princess Tara Zamani
## Date: 12/11/2021

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, depth=3, width=5, netType='Relu'):
        self.width = width
        self.netType = netType

        super(Net, self).__init__()

        self.layers = nn.ModuleDict() 

        self.layers['input'] = nn.Linear(5, self.width)

        for i in range(1, depth):
            self.layers['hidden_'+str(i)] = nn.Linear(self.width,self.width)

        self.layers['output'] = nn.Linear(self.width, 1)
        self.depth = depth 

    # Forward pass through model for prediction
    #   Inputs:
    #       x - input data
    #   Outputs:
    #       x - prediction of model for input example
    def forward(self, x):
        for layer in self.layers:
            if layer != 'output':
                if self.netType == 'Relu':
                    x = torch.relu(self.layers[layer](x.float()))
                else: 
                    x = torch.tanh(self.layers[layer](x.float()))
            else:
                x = self.layers['output'](x)
        return x

    # Initialize weights based on type of network, relu or tanh
    #   Inputs:
    #       m - neural network
    #   Outputs:
    #       None - updates internal network weights
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.netType == 'Relu':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                torch.nn.init.xavier_normal_(m.weight, gain=1.0)

# Fits the model to the data by running through training and testing.
#   Inputs:
#       train_data - data used for training
#       test_data - data used for testing
#       n - number of features in input data
#       optimizer - optimizer for backprop
#       scheduler - scheduler for learning rate update
#       net - neural net model
#   Outputs:
#       prints the training and testing errors of model every 2500 epochs.
def fit_model(train_data, test_data, n, optimizer, scheduler, net):
    T = 5000 # Epochs

    for epoch in range(1,T):
        # Shuffle training set
        shuffled_data = train_data.sample(frac=1) 
        data = shuffled_data.iloc[:, :n].to_numpy()
        labels = torch.tensor(shuffled_data.iloc[:, n].to_numpy())

        shuffled_data_test = test_data.sample(frac=1) 
        data_test = shuffled_data_test.iloc[:, :n].to_numpy()
        labels_test = torch.tensor(shuffled_data_test.iloc[:, n].to_numpy())

        pred = net(torch.tensor(data))
        loss = torch.mean(1/2*torch.square(pred-labels)) 

        # Clear the grads
        optimizer.zero_grad()
        # Backward and build the computational graph
        loss.backward()
        # Updates
        optimizer.step()
        scheduler.step()
        
        if epoch%2500 == 0 or epoch == 4999:
            print(f'Epoch #{epoch}\t: ', end='')
            with torch.no_grad():
                
                rmse_tr = torch.mean(1/2*torch.square(labels-pred))
                rmse_te = torch.mean(1/2*torch.square(labels_test-net(torch.tensor(data_test))))
                
                print('train_err={:.5f}, test_err={:.5f}'.format(rmse_tr, rmse_te.item()))
        

# Load data from .csv file
#   Inputs:
#       fileName - name of .csv file
#       headerNames - list of column header names for the data file
#   Outputs:
#       data - file data in pandas DataFrame format
def load_data(fileName, headerNames):
    data = pd.read_csv(fileName, header=None, names=headerNames)
    return data


def main():
    # Load data -- bank-note
    headerNames = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    n = len(headerNames) # This works because the bias term will be added, hence the number of data columns would be 5 

    train_data = load_data("bank-note/train.csv", headerNames)
    test_data = load_data("bank-note/test.csv", headerNames)

    # Change labels [0, 1] to [-1, 1]
    train_data['label'].loc[train_data['label'] == 0] = -1
    test_data['label'].loc[test_data['label'] == 0] = -1

    # Augment data vectors
    train_data.insert(n-1, 'bias', 1)
    test_data.insert(n-1, 'bias', 1)

    # Loop through widths
    widths = [5,10,25,50,100]
    depths = [3,5,9]

    #### Relu network 
    print("\n ------------------- ReLu Network------------------- ")
    # Loop through depths
    for d in depths:
        # Loop through widths using test function
        for w in widths:
            print(f"#####  Results for (Depth, Width) = ({d}, {w}) #####")
            net = Net(depth=d, width=w, netType='Relu')
            net.apply(net.init_weights)
            net = net.float()

            # create your optimizer 
            optimizer = optim.Adam(net.parameters(), lr=0.0001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99, verbose=False)

            fit_model(train_data, test_data, n, optimizer, scheduler, net)

    #### Tanh network 
    print("\n\n ------------------- Tanh Network ------------------- ")
    # Loop through depths
    for d in depths:
        # Loop through widths using test function
        for w in widths:
            print(f"##### Results for (Depth, Width) = ({d}, {w}) #####")
            net = Net(depth=d, width=w, netType='Tanh')
            net.apply(net.init_weights)
            net = net.float()

            # create your optimizer 
            optimizer = optim.Adam(net.parameters(), lr=0.0001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99, verbose=False)

            fit_model(train_data, test_data, n, optimizer, scheduler, net)









if __name__ == "__main__":
    main()
