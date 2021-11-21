## Machine Learning HW 4 - SVM Implementation for Dual Domain
## Author: Princess Tara Zamani
## Date: 11/17/2021

from re import sub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import warnings
warnings.filterwarnings("ignore")

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
    
    print("************* Dual Domain SVM Results *************")

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

    ####### SVM ########
    C_list = [100/873, 500/873, 700/873]
    C_list_strings = ['100/873', '500/873', '700/873'] # For plotting
    N = len(train_data)

    ## 3a. 
    print('#### Question 3a Results ####')
    # For each setting of C
    for C_idx in range(len(C_list)):
        C = C_list[C_idx]
        print(f'Results for C = {C_list_strings[C_idx]}:')
        # Given a training set S = {(x_i, y_i)}, x_i is in R^n, y_i is in {-1,1}
        # Initializations
        w = np.zeros(n)

        # Shuffle the data
        shuffled_data = train_data.sample(frac=1) 
        sample = shuffled_data.iloc[:, :n].to_numpy()
        label = shuffled_data.iloc[:, n].to_numpy()
        
        subfunc = np.outer(label, label) * np.inner(sample, sample)
        obj_fun = lambda alphas: 1/2*np.sum(np.outer(alphas, alphas)*subfunc) - np.sum(alphas)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x*label)}) 
        res = opt.minimize(obj_fun, x0=np.zeros(shape=len(sample),), method='SLSQP', bounds=opt.Bounds(0,C), constraints=constraints)
        
        opt_alphas = np.array(res.get('x'))
        w = np.sum(sample.T * np.multiply(np.array(opt_alphas), label), axis=1) 
        print(f'W = {w}')
        bias = 1/len(train_data) * np.sum(label - np.sum(np.tile(w, (872,1))*sample, axis=1)) 
        print(f'Bias = {bias}')

        # print(f'Number of support vectors = {np.count_nonzero(opt_alphas)}')

        # Prediction
        preds = np.sign(np.sum(np.tile(w, (872,1))*sample, axis=1) + bias)
        correctNum = np.count_nonzero(preds == label)
        error = 1 - (correctNum / len(train_data))
        print(f'Error = {error}')



if __name__ == "__main__":
    main()