## Machine Learning HW 4 - SVM Implementation for Dual Domain with Gaussian Kernel
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

# Gaussian Kernel 
#   Inputs:
#       Xi - input 1
#       Xj - input 2
#       sigma - variance^2. A free parameter
#   Outputs:
#       gaussian kernal transform 
def gaussian_kernel(Xi, Xj, sigma):
    return np.exp(-(1 / sigma) * np.linalg.norm(Xi[:, np.newaxis] - Xj[np.newaxis, :], axis=2) ** 2)
    # return np.exp(- np.linalg.norm(Xi - Xj)**2 / sigma)


def main():
    
    print("************* Dual Domain SVM Results with Kernel *************")

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
    C_list = [500/873] #[100/873, 500/873, 700/873]
    C_list_strings = ['500/873'] #['100/873', '500/873', '700/873'] # For plotting
    N = len(train_data)
    sigmas = [0.1, 0.5, 1, 5, 100]
    support_vecs = []

    ## 3b. 
    print('#### Question 3b Results ####')
    # For each setting of C
    for C_idx in range(len(C_list)):
        C = C_list[C_idx]
        print(f'Results for C = {C_list_strings[C_idx]}:')
        # Given a training set S = {(x_i, y_i)}, x_i is in R^n, y_i is in {-1,1}
        # Initializations
        w = np.zeros(n)

        for sigma in sigmas:
            # Shuffle the data
            shuffled_data = train_data.sample(frac=1) 
            sample = shuffled_data.iloc[:, :n].to_numpy()
            label = shuffled_data.iloc[:, n].to_numpy()
            
            subfunc = np.outer(label, label) * gaussian_kernel(sample, sample, sigma)
            obj_fun = lambda alphas: 1/2*np.sum(np.outer(alphas, alphas)*subfunc) - np.sum(alphas)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x*label)}) 
            res = opt.minimize(obj_fun, x0=np.zeros(shape=len(sample),), method='SLSQP', bounds=opt.Bounds(0,C), constraints=constraints)
            
            opt_alphas = np.array(res.get('x'))
            inner_calc = np.sum(gaussian_kernel(sample, sample, sigma) * np.multiply(np.array(opt_alphas), label), axis=1)
            bias = 1/len(train_data) * np.sum(label - inner_calc) 
            # print(f'Bias = {bias}')

            print(f'Number of support vectors for C = {C_list_strings[C_idx]} and Sigma = {sigma}: {np.count_nonzero(opt_alphas)}')
            if C_idx == 0:
                support_vecs.append(opt_alphas[np.nonzero(opt_alphas)])
        
        for i in range(len(sigmas)-1):
            print(f'Overlap support vector number between {sigmas[i]} & {sigmas[i+1]}: {np.count_nonzero(np.in1d(support_vecs[i], support_vecs[i+1]))}')


            # Training Prediction
            preds = np.sign(inner_calc + bias)
            correctNum = np.count_nonzero(preds == label)
            error = 1 - (correctNum / len(train_data))
            print(f'Training Error for C = {C_list_strings[C_idx]} and Sigma = {sigma}: {error}')

            # Testing Predictions
            test_data_samples = test_data.iloc[:, :n].to_numpy()
            test_labels = test_data.iloc[:, n].to_numpy()
            subfunc = np.outer(test_labels, test_labels) * gaussian_kernel(test_data_samples, test_data_samples, sigma)
            obj_fun = lambda alphas: 1/2*np.sum(np.outer(alphas, alphas)*subfunc) - np.sum(alphas)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x*test_labels)}) 
            res = opt.minimize(obj_fun, x0=np.zeros(shape=len(test_data_samples),), method='SLSQP', bounds=opt.Bounds(0,C), constraints=constraints)
            
            test_opt_alphas = np.array(res.get('x'))
            test_inner_calc = np.sum(gaussian_kernel(test_data_samples, test_data_samples, sigma) * np.multiply(np.array(test_opt_alphas), test_labels), axis=1)
            test_bias = 1/len(test_data) * np.sum(test_labels - test_inner_calc) 
            test_preds = np.sign(test_inner_calc + test_bias)
            test_correctNum = np.count_nonzero(test_preds == test_labels)
            test_error = 1 - (test_correctNum / len(test_data))
            print(f'Testing Error for C = {C_list_strings[C_idx]} and Sigma = {sigma}: {test_error}')







if __name__ == "__main__":
    main()