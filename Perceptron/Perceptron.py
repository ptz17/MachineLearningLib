## Machine Learning HW 3 - Perceptron Implementation
## Author: Princess Tara Zamani
## Date: 10/30/2021

from os import replace
import pandas as pd
import numpy as np

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
    
    print("************* Standard Perceptron Results *************")

    # Load data -- bank-note
    headerNames = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    train_data = load_data("bank-note/train.csv", headerNames)
    test_data = load_data("bank-note/test.csv", headerNames)

    # Change labels [0, 1] to [-1, 1]
    train_data['label'].loc[train_data['label'] == 0] = -1
    test_data['label'].loc[test_data['label'] == 0] = -1

    # Given a training set D = {(x_i, y_i)}, x_i is in R^n, y_i is in {-1,1}
    # Initialize w = 0 
    n = len(headerNames) # This works because the bias term will be added, hence the number of data columns would be 5 
    w = np.zeros(n)

    # Augment data vectors
    train_data.insert(n, 'bias', 1)
    test_data.insert(n, 'bias', 1)

    # For epoch = 1...T:
    T = 10 # max number = 10
    lr = 1e-3 # learning rate
    for epoch in range(1, T+1):
        # Shuffle the data
        shuffled_data = train_data.sample(frac=1) 
        data = shuffled_data.iloc[:, :n].to_numpy()
        labels = shuffled_data.iloc[:, n].to_numpy()

        # For each training example (x_i, y_i) in D:
        for i in range(len(shuffled_data)):
            sample = data[i,:]
            label = labels[i]

            # Prediction
            y_pred = np.sign(np.sum(np.multiply(w, sample)))
            y_pred = -1 if y_pred == 0.0 else y_pred

            # If y_i*w^T*x_i <= 0, update w <- w + lr*y_i*x_i
            if (label*np.matmul(w,sample) <= 0):  
                w = w + lr*label*sample 

        # Return weight vector, w
        print("W = ", w)

        # Return Average prediction error on test data set
        incorrect_num = 0
        test_data_samples = test_data.iloc[:, :n].to_numpy()
        test_labels = test_data.iloc[:, n].to_numpy()
        for i in range(len(test_data)):
            example = test_data_samples[i,:]
            label = test_labels[i]

            # Prediction
            y_pred = np.sign(np.sum(np.multiply(w, example)))
            y_pred = -1 if y_pred == 0.0 else y_pred

            # Accumulate incorrect predictions
            if y_pred != label:
                incorrect_num = incorrect_num + 1

        avg_error = incorrect_num / len(test_data)
        print("Errors : ", avg_error)
    






if __name__ == "__main__":
    main()