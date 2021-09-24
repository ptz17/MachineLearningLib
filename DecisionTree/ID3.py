## Machine Learning HW 1 - Decision Tree Implementation
## Author: Princess Tara Zamani
## Date: 09/19/2021

import pandas as pd
import numpy as np
import warnings

# Load data from .csv file
#   Inputs:
#       fileName - name of .csv file
#       headerNames - list of column header names for the data file
#   Outputs:
#       data - file data in pandas DataFrame format
def load_data(fileName, headerNames):
    data = pd.read_csv(fileName, header=None, names=headerNames)
    return data

# Calculate the entropy of a set for the given attribute. If the attirbute = 'label', 
# then the entropy of the whole set will be calculated.
#   Inputs:
#       S - set of Examples
#       attribute = the attribute of the set to calc entropy for.
#   Outputs:
#       entropy - expected entropy of set S
def calc_entropy(S, attribute):
    if attribute=='label': # Calculate entropy of the entire set
        entropy_node = 0 
        values = S.label.unique() 
        for value in values:
            fraction = S.label.value_counts()[value]/len(S.label)  
            entropy_node += -fraction*np.log2(fraction)
        return entropy_node

    else: # Calculate the entropy of the set for the given attribute.
        labels = S.label.unique()  
        features = S[attribute].unique()   
        entropy_attribute = 0
        for curr_feature in features:
            entropy_each_feature = 0
            for curr_label in labels:
                num = len(S[attribute][S[attribute]==curr_feature][S.label == curr_label]) 
                den = len(S[attribute][S[attribute]==curr_feature])  
                frac = num/den
                if not frac: # If fraction = 0 --> log(0) gives an error
                    entropy_each_feature = 0
                    break
                else:
                    entropy_each_feature += -frac*np.log2(frac) 
            weight = den/len(S)
            entropy_attribute += weight*entropy_each_feature   # Sums up all the partial entropies

        return entropy_attribute


# Calculate the Majority Error of a set for the given attribute. If the attirbute = 'label', 
# then the ME of the whole set will be calculated.
#   Inputs:
#       S - set of Examples
#       attribute = the attribute of the set to calc entropy for.
#   Outputs:
#       MajError - expected ME of set S
def calc_maj_error(S, attribute):
    if attribute=='label': # Calculate ME of the entire set
        majValueCnt = S.label.value_counts().max()
        totalCnt = len(S.label)
        return 1 - (majValueCnt/totalCnt)

    else: # Calculate the ME of the set for the given attribute.
        features = S[attribute].unique()   
        MErr_attribute = 0
        for feature in features:
            MErr_each_feature = 0
            Sv = S[S[attribute] == feature] # Isolate subset
            majValueCnt = Sv.label.value_counts().max() # Find number of elements in majority
            totalCnt = len(Sv.label)
            MErr_each_feature = 1 - (majValueCnt/totalCnt) 
            frac = len(Sv)/len(S)
            MErr_attribute += frac*MErr_each_feature   # Sums up all the partial MEs

        return MErr_attribute


# Calculate the Gini Index of a set for the given attribute. If the attirbute = 'label', 
# then the GI of the whole set will be calculated.
#   Inputs:
#       S - set of Examples
#       attribute = the attribute of the set to calc entropy for.
#   Outputs:
#       GI - expected Gini Index of set S
def calc_GI(S, attribute):
    if attribute=='label': # Calculate GI of the entire set
        GiniIdx_Sum = 0  
        labels = S.label.unique()  
        for label in labels:
            fraction = S.label.value_counts()[label]/len(S.label)  
            GiniIdx_Sum += fraction**2
        return 1 - GiniIdx_Sum

    else: # Calculate the GI of the set for the given attribute.
        labels = S.label.unique()  
        features = S[attribute].unique() 
        GiniIdx_attribute = 0
        for feature in features:
            GiniIdxSum_each_feature = 0
            for label in labels:
                num = len(S[attribute][S[attribute]==feature][S.label == label]) 
                den = len(S[attribute][S[attribute]==feature])  
                frac = num/(den)
                GiniIdxSum_each_feature += frac**2 
            weight = den/len(S)
            GiniIdx_attribute += weight*(1-GiniIdxSum_each_feature)   # Sums up all the partial GISums 

        return GiniIdx_attribute


# Gets the best attribute for splitting the set on. The user can specify which heuristic 
# method to use (entropy, majority error, or gini index) by input.
#   Inputs:
#       S - set of Examples
#       Attr - the attribute of the set to calc entropy for.
#       Heuristic - specification of the heuristic type used to choose best attribute
#   Outputs:
#       A - the best attribute to split set on
def get_best_attr(S, Attr, Heuristic):
    Atrr_keys = list(Attr)

    if Heuristic == 'Entropy':
        curr_entropy = calc_entropy(S, 'label')
        info_gains = np.zeros(len(Atrr_keys))
        for i in range(len(Atrr_keys)):
            attr_entropy = calc_entropy(S, Atrr_keys[i])
            info_gains[i] = curr_entropy - attr_entropy
    
    elif Heuristic == 'Majority Error':
        curr_ME = calc_maj_error(S, 'label')
        info_gains = np.zeros(len(Atrr_keys))
        for i in range(len(Atrr_keys)):
            attr_ME = calc_maj_error(S, Atrr_keys[i])
            info_gains[i] = curr_ME - attr_ME

    elif Heuristic == 'Gini Index':
        curr_GI = calc_GI(S, 'label')
        info_gains = np.zeros(len(Atrr_keys))
        for i in range(len(Atrr_keys)):
            attr_GI = calc_GI(S, Atrr_keys[i])
            info_gains[i] = curr_GI - attr_GI

    else:
        print('Incorrect Heuristic Listed')

    A = Atrr_keys[info_gains.argmax()]
    return A


# Implement the ID3 algorithm that supports information gain, majority error, and gini index 
# to select attributes for data spilts. Allow user to set the maximum tree depth.
#   Inputs:
#       S - set of Examples
#       Attr - set of measured attributes
#       label - target attribute (the prediction)
#       MaxDepth - the maximum depth inputted by user. 
#       CurrentDepth - current depth of tree.
#       HeurisitcChoice - user's choice of heurisitc for deciding splitting features
#   Outputs:
#       root - node of tree with its acquired branches
def ID3(S, Attr, MaxDepth, CurrentDepth=0, HeuristicChoice='Entropy'):
    labels, cnts = np.unique(S['label'], return_counts=True)
    if (len(cnts) == 1): # If all examples have the same label --> return leaf node with the label
        return labels[0]
    elif not Attr or (CurrentDepth == MaxDepth): # If attributes is empty or tree has reached max depth --> return leaf node with most common label
        mostCommlabel = S['label'].value_counts().idxmax()
        return mostCommlabel
    else:
        # 1. Create a root node for tree
        root = {}

        # 2. A = attribute in Attr that best splits S
        A = get_best_attr(S, Attr, HeuristicChoice)
        root[A] = {}

        # 3. for each possible value v of that A can take:
        # Allow for numerical attributes to have binary tree structure
        if Attr[A] == 'numeric':
            possibleValues = [0, 1] 
        else:
            possibleValues = Attr[A]

        for v in possibleValues: 
            #   1. Add a new tree branch corresponding to A = v
            root[A][v] = {}
            #   2. Let Sv be the subset of examples in S with A = v
            Sv = S[S[A] == v]
            #   3. if Sv is empty --> add leaf node with most common value of label in S
            if Sv.empty:
                root[A][v] = S['label'].value_counts().idxmax()
            #      else --> below this branch, add the subtree ID3(Sv, Attr - {A}, label)
            else:
                newAtrr = Attr.copy()
                newAtrr.pop(A)
                root[A][v] = ID3(Sv, newAtrr, MaxDepth, CurrentDepth+1)

        # 4. Return root node 
        return root


# Will use the given tree to make a prediction based on input example.
#   Inputs:
#       tree - the learned tree 
#       example - example to make prediction of
#   Outputs:
#       pred - prediction based on following tree.
def tree_prediction(tree, example): 
    for key, value in tree.items():
        subtree = value[example[key]]

        if isinstance(subtree, str):
            return subtree
        else:
            pred = tree_prediction(subtree, example)
            if isinstance(pred, str):
                return pred


# Recursively finds the depth of the given dic.
# Inputs:
#       dic - input dictionary
#       depth - current depth of the tree
# Outputs:
#       depth - returns the depth of the tree
def dict_depth(dic, depth = 1):
    if not isinstance(dic, dict) or not dic:
        return depth
    return max(dict_depth(dic[key], depth + 1)
                               for key in dic)


def main():
    ##### Question 2
    # Load data -- car
    headerNames =['buying','maint','doors','persons','lug_boot','safety','label']
    train_data = load_data("car/train.csv", headerNames)
    test_data = load_data("car/test.csv", headerNames)
    attributes = {"buying":['vhigh', 'high', 'med', 'low'], "maint":['vhigh', 'high', 'med', 'low'], "doors":['2', '3', '4', '5more'],
                    "persons":['2', '4', 'more'], "lug_boot":['small', 'med', 'big'], "safety":['low', 'med', 'high']}
    labels = train_data['label'] 

    # 2b
    print("Question 2(b) - Car Dataset")
    # Loop through types of Heuristics
    heurisitcs = ['Entropy', 'Majority Error', 'Gini Index']
    for heuristic in heurisitcs:
        train_err_sum = 0
        test_err_sum = 0
        avg_train_err = 0
        avg_test_err = 0

        # Vary maximum tree depth from 1 to 6
        for maxDepth in range(1,7):
            # For each setting, 
            # Run algorithm to learn a decision tree
            tree = ID3(train_data, attributes, maxDepth, 0, heuristic)
            depth = int(dict_depth(tree)/2) # tree dictionary includes attribute values in depth. Those account for half of the depth.

            if depth < maxDepth:
                break

            # Use tree to preditct both the training and test examples.
            # Training examples
            correct_pred = 0
            for i in range(len(train_data)):
                pred = tree_prediction(tree, train_data.iloc[i])
                if (pred == labels[i]):
                    correct_pred += 1
            train_pred_error = 1 - correct_pred/len(train_data)
            train_err_sum += train_pred_error

            # Test examples
            correct_pred = 0
            for i in range(len(test_data)):
                pred = tree_prediction(tree, test_data.iloc[i])
                if (pred == test_data['label'][i]):
                    correct_pred += 1
            test_pred_error = 1 - correct_pred/len(test_data)
            test_err_sum += test_pred_error

        # Report the average prediction errors on each data set when you use information gain, majority error, and gini index heuristics
        avg_train_err = train_err_sum/depth
        avg_test_err = test_err_sum/depth 
        print("Average Train Error for ", heuristic, ": ", avg_train_err)
        print("Average Test Error for ", heuristic, ": ", avg_test_err)



    ##### Question 3
    # Load data -- bank
    attributes = {"age":'numeric', 
                "job":["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"], 
                "marital":["married","divorced","single"],
                "education":["unknown","secondary","primary","tertiary"], 
                "default":["yes", "no"], 
                "balance":'numeric',
                "housing":["yes", "no"], 
                "loan":["yes", "no"],
                "contact":["unknown","telephone","cellular"],
                "day":'numeric',
                "month":["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                "duration":'numeric',
                "campaign":'numeric',
                "pdays":'numeric',
                "previous":'numeric',
                "poutcome":["unknown","other","failure","success"]}
    headerNames = list(attributes.keys())
    headerNames.append('label')

    # Create list of numerical attributes
    attributeVals = np.array(list(attributes.values()), dtype=object)
    numericIndices = np.where(attributeVals == 'numeric')[0]
    numericAttributes = np.array(headerNames)[numericIndices]

    train_data = load_data("bank/train.csv", headerNames)
    test_data = load_data("bank/test.csv", headerNames)
    labels = train_data['label']

    # Change numeric attributes to binary ones.
    for a in numericAttributes:
        media = np.median(train_data[a])
        train_data[a] = (train_data[a] >= media).astype(int)
        test_data[a] = (test_data[a] >= media).astype(int)


    ## 3a
    print("Question 3(a) - Bank Dataset")
    # Loop through types of Heuristics
    for heuristic in heurisitcs:
        train_err_sum = 0
        test_err_sum = 0
        avg_train_err = 0
        avg_test_err = 0

        # Vary maximum tree depth from 1 to 16
        for maxDepth in range(1,17):
            # For each setting, 
            # Run algorithm to learn a decision tree
            tree= ID3(train_data, attributes, maxDepth, 0, heuristic)
            depth = int(dict_depth(tree)/2) # tree dictionary includes attribute values in depth. Those account for half of the depth.

            if depth < maxDepth:
                break

            # Use tree to preditct both the training and test examples.
            # Training examples
            correct_pred = 0
            for i in range(len(train_data)):
                pred = tree_prediction(tree, train_data.iloc[i])
                if (pred == labels[i]):
                    correct_pred += 1
            train_pred_error = 1 - correct_pred/len(train_data)
            train_err_sum += train_pred_error
            # print("Train Errors")
            # print(train_pred_error)

            # Test examples
            correct_pred = 0
            for i in range(len(test_data)):
                pred = tree_prediction(tree, test_data.iloc[i])
                if (pred == test_data['label'][i]):
                    correct_pred += 1
            test_pred_error = 1 - correct_pred/len(test_data)
            test_err_sum += test_pred_error
            # print("Test Errors")
            # print(test_pred_error)

        # Report the average prediction errors on each data set when you use information gain, majority error, and gini index heuristics
        avg_train_err = train_err_sum/depth
        avg_test_err = test_err_sum/depth
        print("Average Train Error for ", heuristic, ": ", avg_train_err)
        print("Average Test Error for ", heuristic, ": ", avg_test_err)


    ## 3b
    print("Question 3(b) - Bank Dataset")
    # Replace "unknown" attributes with the majority of other values of the same attribute in the training set
    warnings.simplefilter(action='ignore', category=FutureWarning) # for line 389
    for a in headerNames[:len(headerNames)-1]:
        if 'unknown' in train_data[a].unique():
            value_cnts = train_data[a].value_counts().drop(labels='unknown')
            majority_val = value_cnts.idxmax()
            train_data[a] = train_data[a].replace('unknown', majority_val)

    # Loop through types of Heuristics
    for heuristic in heurisitcs:
        train_err_sum = 0
        test_err_sum = 0
        avg_train_err = 0
        avg_test_err = 0

        # Vary maximum tree depth from 1 to 16
        for maxDepth in range(1,17):
            # For each setting, 
            # Run algorithm to learn a decision tree
            tree= ID3(train_data, attributes, maxDepth, 0, heuristic)
            depth = int(dict_depth(tree)/2) # tree dictionary includes attribute values in depth. Those account for half of the depth.

            if depth < maxDepth:
                break

            # Use tree to preditct both the training and test examples.
            # Training examples
            correct_pred = 0
            for i in range(len(train_data)):
                pred = tree_prediction(tree, train_data.iloc[i])
                if (pred == labels[i]):
                    correct_pred += 1
            train_pred_error = 1 - correct_pred/len(train_data)
            train_err_sum += train_pred_error
            # print("Train Errors")
            # print(train_pred_error)

            # Test examples
            correct_pred = 0
            for i in range(len(test_data)):
                pred = tree_prediction(tree, test_data.iloc[i])
                if (pred == test_data['label'][i]):
                    correct_pred += 1
            test_pred_error = 1 - correct_pred/len(test_data)
            test_err_sum += test_pred_error
            # print("Test Errors")
            # print(test_pred_error)

        # Report the average prediction errors on each data set when you use information gain, majority error, and gini index heuristics
        avg_train_err = train_err_sum/depth
        avg_test_err = test_err_sum/depth
        print("Average Train Error for ", heuristic, ": ", avg_train_err)
        print("Average Test Error for ", heuristic, ": ", avg_test_err)



if __name__ == "__main__":
    main()