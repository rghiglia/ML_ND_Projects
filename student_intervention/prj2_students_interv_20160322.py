# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:35:25 2016

@author: rghiglia
"""

# prj2_students_interv_20160322

import numpy as np
import pandas as pd
from pandas import DataFrame

# Load data
student_data = pd.read_csv(r"C:\Users\rghiglia\Documents\ML_ND\student_intervention\student-data.csv")
type(student_data)  # dataframe

nO, nF = student_data.shape
col_nm = student_data.columns.tolist()
pss = student_data[col_nm[-1]]

isPass = np.array(pss=='yes')
nPass = sum(isPass==True)
nFail = sum(isPass==False)
grad = float(nPass) / float(nO)

feat_nm = col_nm[:-1]
labl_nm = col_nm[-1]

# Some features assume only two values, e.g. yes/no, etc.
# Those can be converted into 0/1

# Some have multiple values and are called categorical
# The recommended way to convert them is to generate 0/1 for each value of the category

# How would I do this?
# Take a column that is non-numeric
# Take for value (= val1) and create a column name = Var_val1
# Assign a 1 for that entry
# Loop to next value
# If you encounter a new value, create an extra column = Var_val2, initiate to 0 for all entries up to this point (necessarily true otherwise you'd have encountered already)

def Category2Cols(df):
    # Input a 1 column dataframe
    col = df.columns.tolist()[0]
    vals = df[col]
    nR = len(vals)
    df_aug = DataFrame(np.zeros(nR),columns=[col + "_" + vals[0]])
    cols = [col + "_" + vals[0]]    # initializes to a list
    df_aug.loc[0,cols[-1]] = 1
    for i in range(1,nR):
        print i, vals[i]
        if vals[i] in cols:
           df_aug.loc[i,[[col + "_" + vals[i]]]] = 1
        else:
           cols.append(col + "_" + vals[i])
           df_aug.loc[i,cols[-1]] = 1
           df_aug.loc[np.isnan(df_aug.loc[:,cols[-1]]),cols[-1]] = 0
    
    return df_aug

#df_aug = Category2Cols(pd.DataFrame({"Mjob", student_data["Mjob"]})
#df = DataFrame({"Mjob", student_data["Mjob"]}) # nope
#type(df) # nope! returns a Series
#type(student_data) # a DataFrame

## Defining a dataframe
#>>> d = {'col1': ts1, 'col2': ts2}
#>>> df = DataFrame(data=d, index=index)
#>>> df2 = DataFrame(np.random.randn(10, 5))
#>>> df3 = DataFrame(np.random.randn(10, 5),
#...                 columns=['a', 'b', 'c', 'd', 'e'])
df = DataFrame(data=student_data["Mjob"], columns=['Mjob']) # ok
col = df.columns.tolist()[0]
df_aug = Category2Cols(df)

# Nice
# Probably not most efficient but gets the job done

# Other version:
# Extract uniques by set(vals)
# Create 


def Category2Cols2(df):
    # Input a 1 column dataframe
    col = df.columns.tolist()[0]
    vals, valU = df[col], list(set(df[col]))
    nR, nF = len(vals), len(valU)
    cols = []
    for i in range(nF):
        cols.append(col + "_" + valU[i])
    df_aug = DataFrame(np.zeros((nR,nF)),columns=cols)
    for i in range(nR):
        df_aug.loc[i,col + "_" + vals[i]] = 1
    
    return df_aug

df_aug2 = Category2Cols2(df)
# Cleaner, not sure in Pythonian style, but ...

# Next step: extract data types
df_all = []
for (col, dt) in zip(student_data.columns.tolist(), student_data.dtypes):
    print col, dt
    df = DataFrame(data=student_data[col], columns=[col]) # ok
    if dt!=np.int64:
        df_aug = Category2Cols2(df)
        df_all.append(df_aug)
    else:
        df_all.append(df)

# Awesome, only problem now that I have a list of dataframes ...
df_all2 = pd.concat(df_all,axis=1)

# Friggin' amazing!

# Geez, there is already a function that does that!!!
# pd.get_dummies()

# Note: my version creates redundant features though: in the case of 2 values
# you still have two columns which are the exact opposite

## Review code and use get_dummies()
#df_all3 = 


# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX


# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
y_all_d = np.array(y_all.replace(['yes', 'no'], [1, 0]))
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

# Friggin' amazing
X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
from sklearn.utils import shuffle
X_all, y_all, y_all_d = shuffle(X_all, y_all, y_all_d, random_state=0)

# Convert targets into 0s and 1s
X_train, y_train = X_all[:num_train,:], y_all_d[:num_train]
X_test, y_test = X_all[num_train:,:], y_all_d[num_train:]

print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data

# Train a model
import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# Predict on training set and compute F1 score
from sklearn.metrics import f1_score, precision_score, recall_score

# Note: F1_score is presumably for classification problems

def bool2YN(x):
    y = []
    for xi in x:
        if xi==0:
            y.append('no')
        else:
            y.append('yes')
    return y

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    if clf.__class__.__name__=='LinearRegression':
        y_t = bool2YN(target)
        y_p = bool2YN(np.maximum(np.minimum(np.round(y_pred),1),0))
        p = precision_score(y_t, y_p, pos_label='yes')
        r = recall_score(y_t, y_p, pos_label='yes')
        print "Precision = ", p
        print "Recall = ", r
        return f1_score(y_t, y_p, pos_label='yes')
    else:
#        y_t = bool2YN(target)
#        y_p = bool2YN(np.maximum(np.minimum(np.round(y_pred),1),0))
#        p = precision_score(y_t, y_p, pos_label='yes')
#        r = recall_score(y_t, y_p, pos_label='yes')
#        print "Precision = ", p
#        print "Recall = ", r
#        return f1_score(target.values, y_pred, pos_label='yes')
        return f1_score(target, y_pred, pos_label='yes')

# TODO: Choose a model, import it and instantiate an object

# Linear Regression
from sklearn import linear_model
clf = linear_model.LinearRegression(fit_intercept=True, normalize=False)
train_classifier(clf, X_train, y_train)
train_f1_score_LR = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score_LR)

## Checks
#y_pred = clf.predict(X_train)
#y_p = bool2YN(np.maximum(np.minimum(np.round(y_pred),1),0))
#y_t = bool2YN(y_train)
#f1_score(y_t, y_p, pos_label='yes')
#p = precision_score(y_t, y_p, pos_label='yes')
#r = recall_score(y_t, y_p, pos_label='yes')

# Recall: ability to get positives
# Precison: not to screw up with false negatives
# F1: weighted average of precision and recall


# Support vector machine
X_train, y_train = X_all[:num_train,:], y_all[:num_train]
X_test, y_test = X_all[num_train:,:], y_all[num_train:]

from sklearn import svm
clf = svm.SVC()
train_classifier(clf, X_train, y_train)
train_f1_score_SVM = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score_SVM)


# Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
train_classifier(clf, X_train, y_train)
train_f1_score_kNN = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score_kNN)


# Decision tree
from sklearn.tree import DecisionTreeClassifier
# what's the decision tree regressor?
clf = DecisionTreeClassifier(max_depth=5)
train_classifier(clf, X_train, y_train)
train_f1_score_DT = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score_DT)
# Ulla ... F1 = 1 ..., ok I think it had to do with max_depth
#f1_score(target, y_pred, pos_label='yes')
# Export tree
from sklearn.externals.six import StringIO

# Damn! I still can't visualize the tree
import sys
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\pydot-1.0.28')
del sys.path[-1]
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
pydot.graph_from_dot_data(dot_data.getvalue())

