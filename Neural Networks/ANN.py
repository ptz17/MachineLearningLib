
## Machine Learning HW 5 - ANN implementation
## Author: Princess Tara Zamani
## Date: 12/11/2021

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, width=3, weights_initialize='Gaussian'):
        self.width = width

        if weights_initialize=='BackPropTest':
            # Test weights 
            self.w_hidden1 = np.array([[-1, 1], [-2, 2], [-3, 3]], dtype=float) # input to hidden layer 1 weights
            self.w_hidden2 = np.array([[-1, 1], [-2, 2], [-3, 3]], dtype=float) # hidden layer 1 to hidden layer 2 weights
            self.w_outputlayer = np.array([[-1], [2], [-1.5]])                  # hidden layer 2 to output weights
        elif weights_initialize=='Gaussian':
            self.w_hidden1 = np.random.randn(5,self.width-1)            # input to hidden layer 1 weights
            self.w_hidden2 = np.random.randn(self.width,self.width-1)   # hidden layer 1 to hidden layer 2 weights
            self.w_outputlayer = np.random.randn(self.width,1)          # hidden layer 2 to output weights
        elif weights_initialize=='Zeros':
            self.w_hidden1 = np.zeros((5,self.width-1))                 # input to hidden layer 1 weights
            self.w_hidden2 = np.zeros((self.width,self.width-1))        # hidden layer 1 to hidden layer 2 weights
            self.w_outputlayer = np.zeros((self.width,1))               # hidden layer 2 to output weights
        
    # Sigmoid function
    #   Inputs:
    #       x - input data
    #       w - input weights
    #   Outputs:
    #       Sigmoid result
    def sigmoid(self, x, w):
        z = np.dot(x, w)
        return 1/(1 + np.exp(-z))
    
    # Derivative of Sigmoid function
    #   Inputs:
    #       x - input data
    #       w - input weights
    #   Outputs:
    #       Derivate of Sigmoid result
    def sigmoid_derivative(self, x, w):
        return self.sigmoid(x, w) * (1 - self.sigmoid(x, w))

    # Forward pass through model for prediction
    #   Inputs:
    #       x - input data
    #   Outputs:
    #       hidden1 - first hidden layer. Preserved for later use in backprop
    #       hidden2 - second hidden layer. Preserved for later use in backprop
    #       y_pred - prediction of model for input example
    def forward(self, x):
        hidden1 = np.concatenate(([1], self.sigmoid(x, self.w_hidden1)))        # reconcatenate with bias term
        hidden2 = np.concatenate(([1], self.sigmoid(hidden1, self.w_hidden2)))  # reconcatenate with bias term
        y_pred = hidden2.dot(self.w_outputlayer)
        # print("forward y_pred = ", y_pred)
        return hidden1, hidden2, y_pred

    # Backward pass through model for weight updates
    #   Inputs:
    #       hidden1 - first hidden layer
    #       hidden2 - second hidden layer
    #       x - input data 
    #       y_true - actual label
    #       y_pred - prediction of model for input example
    #       lr - learning rate
    #   Outputs:
    #       Weights of the three layers in the network: w_hidden1, w_hidden2, w_outputLayer
    def backward(self, hidden1, hidden2, x, y_true,y_pred, lr=1e-4, test=False):
        loss_grad = y_pred - y_true
        grad_w3 = loss_grad * hidden2

        cache = np.multiply(self.w_outputlayer[1:].T, (loss_grad * self.sigmoid_derivative(hidden1, self.w_hidden2)).reshape(1,self.width-1))
        grad_w2 = hidden1.reshape(self.width,1) * cache

        multipliers = np.multiply(cache.T, self.w_hidden2[1:].T)
        grad_w1 = np.zeros((len(x),self.width-1))
        for i in range(len(x)): # Number of inputs
            temp = x[i] * multipliers
            temp = np.sum(temp, axis=0)
            grad_w1[i] = temp
        

        if test:
            print("\nLayer 3 weights: \n", grad_w3)
            print("\nLayer 2 weights: \n", grad_w2)
            print("\nLayer 1 weights: \n", grad_w1)

        # Update and return new weights for the 3 layers
        self.w_hidden1 -= lr * grad_w1
        self.w_hidden2 -= lr * grad_w2
        self.w_outputlayer -= (lr * grad_w3).reshape(self.width,1)
        return self.w_hidden1, self.w_hidden2, self.w_outputlayer

    # Stochastic Gradient Descent algorithm
    #   Inputs:
    #       n - number of features in input data
    #       train_data - data samples from training set
    #   Outputs:
    #       Prints the final weights or error found for the current width of network
    def SGD(self, train_data, n):
        lr_0 = 1e-3
        d = 0.009 
        T = 100 # Epochs
        y_points = []
        updateNum = 0

        train_errors = []

        for epoch in range(1,T):
            # Shuffle training set
            shuffled_data = train_data.sample(frac=1) 
            data = shuffled_data.iloc[:, :n].to_numpy()
            labels = shuffled_data.iloc[:, n].to_numpy()

            incorrect_num = 0
            loss = 0
            # For each example...
            for t in range(len(shuffled_data)):
                # Obtain example t and its label
                sample = data[t,:]
                label = labels[t]

                # Update learning rate
                lr = lr_0 / (1 + lr_0*t/d)

                # Treat the example as the entire dataset. Compute the gradient of the loss using backward. Update w (in backward)
                H1, H2, y_pred = self.forward(sample)
                loss += 1/2 * np.power((label - y_pred),2)
                self.backward(H1, H2, sample, label, y_pred, lr=lr)

    
            # Accumulate average training errorss
            train_errors.append(loss/len(train_data))
            
            y_points.append(y_pred[0])
            updateNum += 1

        print(f"Training Error for Width = {self.width}: \t{train_errors[-1]}")
        # print(f"Training Error for Width = {self.width}: \t{np.mean(train_errors)}")

        # print(f"Final Weights for Width = {self.width}")
        # print("\nLayer 1 weights: \n", self.w_hidden1)
        # print("\nLayer 2 weights: \n", self.w_hidden2)
        # print("\nLayer 3 weights: \n", self.w_outputlayer)


    # Runs network through testing dataset
    #   Inputs:
    #       n - number of features in input data
    #       test_data - data samples from testing set
    #   Outputs:
    #       Prints the final weights or error found for the current width of network
    def test(self, test_data, n):
            T = 100 # Epochs
            y_points = []
            updateNum = 0

            test_errors = []

            for epoch in range(1,T):
                # Shuffle training set
                shuffled_data = test_data.sample(frac=1) 
                data = shuffled_data.iloc[:, :n].to_numpy()
                labels = shuffled_data.iloc[:, n].to_numpy()

                loss = 0
                # For each example...
                for t in range(len(shuffled_data)):
                    # Obtain example t and its label
                    sample = data[t,:]
                    label = labels[t]

                   # Treat the example as the entire dataset. Compute the gradient of the loss using backward. Update w (in backward)
                    H1, H2, y_pred = self.forward(sample)
                    loss += 1/2 * np.power((label - y_pred),2)
                    
                # Accumulate average training errors 
                test_errors.append(loss/len(test_data))
                
                y_points.append(y_pred[0])
                updateNum += 1

            print(f"Testing Error for Width = {self.width}: \t{test_errors[-1]}")
            # print(f"Testing Error for Width = {self.width}: \t{np.mean(test_errors)}")

            # print(f"Final Weights for Width = {self.width}")
            # print("\nLayer 1 weights: \n", self.w_hidden1)
            # print("\nLayer 2 weights: \n", self.w_hidden2)
            # print("\nLayer 3 weights: \n", self.w_outputlayer)


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


    # Construct Neural Network and test backpropagation
    print('\n********** Testing Back Propagation **********')
    NN = NeuralNetwork(width=3, weights_initialize='BackPropTest')
    x = np.array([1, 1, 1])
    y = np.array([1])
    H1, H2, y_pred = NN.forward(x)
    NN.backward(H1, H2, x, y, y_pred, test=True)

    # Loop through widths
    widths = [5,10,25,50,100]
    print('\n********** Using Gaussian Weights **********')
    for w in widths:
        # SGD
        NN = NeuralNetwork(width=w, weights_initialize='Gaussian')
        NN.SGD(train_data, n)
        NN.test(test_data, n)

    print('\n********** Using Zeroed Weights **********')
    for w in widths:
        # SGD
        NN = NeuralNetwork(width=w, weights_initialize='Zeros')
        NN.SGD(train_data, n)
        NN.test(test_data, n)






if __name__ == "__main__":
    main()
