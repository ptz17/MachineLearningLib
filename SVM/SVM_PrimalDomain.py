## Machine Learning HW 4 - SVM Implementation for Primal Domain
## Author: Princess Tara Zamani
## Date: 11/17/2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    
    print("************* Primal Domain SVM Results *************")

    # Load data -- bank-note
    headerNames = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    n = len(headerNames) # This works because the bias term will be added, hence the number of data columns would be 5 

    train_data = load_data("bank-note/train.csv", headerNames)
    test_data = load_data("bank-note/test.csv", headerNames)

    # Change labels [0, 1] to [-1, 1]
    train_data['label'].loc[train_data['label'] == 0] = -1
    test_data['label'].loc[test_data['label'] == 0] = -1

    # Augment data vectors
    train_data.insert(n, 'bias', 1)
    test_data.insert(n, 'bias', 1)

    ####### SVM ########
    C_list = [100/873, 500/873, 700/873]
    C_list_strings = ['100/873', '500/873', '700/873'] # For plotting
    N = len(train_data)

    print('#### Question 2a Results ####')
    ## 2a. Use schedule of learning rate: lr_t = lr_0 / (1+lr_0*t/a)
    # Learning rate inital parameters -- tune to ensure convergance
    lr_0 = 1e-6
    a = 0.00873

    # For each setting of C
    for C_idx in range(len(C_list)):
        C = C_list[C_idx]
        # Given a training set S = {(x_i, y_i)}, x_i is in R^n, y_i is in {-1,1}
        # Initializations
        w = np.zeros(n)
        train_errors = []
        test_errors = []

        # For epoch = 1...T:
        T = 100
        for epoch in range(1, T+1):
            # Shuffle the data
            shuffled_data = train_data.sample(frac=1) 
            data = shuffled_data.iloc[:, :n].to_numpy()
            labels = shuffled_data.iloc[:, n].to_numpy()

            incorrect_num = 0
            # For each training example (x_i, y_i) in S:
            for t in range(len(shuffled_data)):
                # Obtain example t and its label
                sample = data[t,:]
                label = labels[t]

                # Update learning rate
                lr = lr_0 / (1 + lr_0*t/a)

                # If y_i*w^T*x_i <= 1, update w <- w - lr*y_i*x_i
                if label*np.matmul(w.T, sample) <= 1:  
                    w = w - lr*np.concatenate((w[:-1], np.array([0]))) + lr*C*N*label*sample 
                else:
                    w[:-1] = (1-lr)*w[:-1]

                # Prediction
                y_pred = np.sign(np.sum(np.multiply(w, sample)))
                y_pred = -1 if y_pred == 0.0 else y_pred

                # Accumulate incorrect predictions for error
                if y_pred != label:
                    incorrect_num = incorrect_num + 1

            # Accumulate average training errors 
            avg_train_error = incorrect_num / len(train_data)
            train_errors.append(avg_train_error)
            # print("Train Error : ", avg_train_error)


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

                # Accumulate incorrect predictions for errors
                if y_pred != label:
                    incorrect_num = incorrect_num + 1

            # Accumulate average testing errors 
            avg_test_error = incorrect_num / len(test_data)
            test_errors.append(avg_test_error)
            # print("Test Error : ", avg_test_error)

        # Print w
        print(f'W = {w}')
        # Print Final Errors per C
        print(f'Final Train Error for C = {C_list_strings[C_idx]}:\t {avg_train_error}')
        print(f'Final Test  Error for C = {C_list_strings[C_idx]}:\t {avg_test_error}')
        
        # Plot Training and Testing Errors for each C
        plt.figure(1)
        plt.plot(test_errors, label='C = '+C_list_strings[C_idx])
        plt.title('Q2a. Primal Domain SVM Test Errors')
        plt.xlabel('Epoch Number')
        plt.ylabel('Error')
        plt.legend()

        plt.figure(2)
        plt.plot(train_errors, label='C = '+C_list_strings[C_idx])
        plt.title('Q2a. Primal Domain SVM Train Errors')
        plt.xlabel('Epoch Number')
        plt.ylabel('Error')
        plt.legend()

    # plt.show()







    print('#### Question 2b Results ####')
    ## 2b. Use schedule of learning rate: lr_t = lr_0 / (1+t)
    # Learning rate inital parameters -- tune to ensure convergance
    lr_0 = 1e-4
    
    # For each setting of C
    # C_list = [100/872]
    for C_idx in range(len(C_list)):
        C = C_list[C_idx]
        # Given a training set S = {(x_i, y_i)}, x_i is in R^n, y_i is in {-1,1}
        # Initializations
        w = np.zeros(n)
        train_errors = []
        test_errors = []

        # For epoch = 1...T:
        T = 100
        for epoch in range(1, T+1):
            # Shuffle the data
            shuffled_data = train_data.sample(frac=1) 
            data = shuffled_data.iloc[:, :n].to_numpy()
            labels = shuffled_data.iloc[:, n].to_numpy()

            incorrect_num = 0
            # For each training example (x_i, y_i) in S:
            for t in range(len(shuffled_data)):
                # Obtain example t and its label
                sample = data[t,:]
                label = labels[t]

                # Update learning rate
                lr = lr_0 / (1 + t)

                # If y_i*w^T*x_i <= 1, update w <- w - lr*y_i*x_i
                if label*np.matmul(w.T, sample) <= 1:  
                    w = w - lr*np.concatenate((w[:-1], np.array([0]))) + lr*C*N*label*sample 
                else:
                    w[:-1] = (1-lr)*w[:-1]

                # Prediction
                y_pred = np.sign(np.sum(np.multiply(w, sample)))
                y_pred = -1 if y_pred == 0.0 else y_pred

                # Accumulate incorrect predictions for error
                if y_pred != label:
                    incorrect_num = incorrect_num + 1

            # Accumulate average training errors 
            avg_train_error = incorrect_num / len(train_data)
            train_errors.append(avg_train_error)
            # print("Train Error : ", avg_train_error)


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

                # Accumulate incorrect predictions for errors
                if y_pred != label:
                    incorrect_num = incorrect_num + 1

            # Accumulate average testing errors 
            avg_test_error = incorrect_num / len(test_data)
            test_errors.append(avg_test_error)
            # print("Test Error : ", avg_test_error)
        
        # Print w
        print(f'W = {w}')
        # Print Final Errors per C
        print(f'Final Train Error for C = {C_list_strings[C_idx]}:\t {avg_train_error}')
        print(f'Final Test  Error for C = {C_list_strings[C_idx]}:\t {avg_test_error}')

        # Plot Training and Testing Errors for each C
        plt.figure(3)
        plt.plot(test_errors, label='C = '+C_list_strings[C_idx])
        plt.title('Q2b. Primal Domain SVM Test Errors')
        plt.xlabel('Epoch Number')
        plt.ylabel('Error')
        plt.legend()

        plt.figure(4)
        plt.plot(train_errors, label='C = '+C_list_strings[C_idx])
        plt.title('Q2b. Primal Domain SVM Train Errors')
        plt.xlabel('Epoch Number')
        plt.ylabel('Error')
        plt.legend()
        
    plt.show()
        
            






if __name__ == "__main__":
    main()