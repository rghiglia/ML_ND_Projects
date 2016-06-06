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
y_all_d = np.array(y_all.replace(['yes', 'no'], [1, 0])) # convert targets into 0s and 1s
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

# Friggin' amazing
X_all_aug = preprocess_features(X_all) # careful: this converts the data into and array
print "Processed feature columns ({}):-\n{}".format(len(X_all_aug.columns), list(X_all_aug.columns))
nO_aug, nF_aug = X_all_aug.shape
feat_nm_aug = list(X_all_aug.columns.values)

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
from sklearn.utils import shuffle
X_all_aug_d, y_all, y_all_d = shuffle(X_all_aug, y_all, y_all_d, random_state=0)
# Careful: it converted X_all_aug into an 2D array
type(X_all_aug_d)   # nd array
type(y_all)         # nd array of yes, no
type(y_all_d)       # nd array of 1s and 0s


# Split data set
X_train_df = X_all_aug[:num_train,:]
# Error: unhashable type, so I cannot select a subset (slice) of a dataframe?
X_train_df = X_all_aug.iloc[:num_train]

X_train, y_train, y_train_d = X_all_aug_d[:num_train,:], y_all[:num_train], y_all_d[:num_train]
X_test, y_test, y_test_d = X_all_aug_d[num_train:,:], y_all[num_train:], y_all_d[num_train:]
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data


# Looking at features

# Almost all features are categorical, except age, and number of absences


import matplotlib.pyplot as pl
fig = pl.figure(figsize=(10,80))

# Histogram
for i in range(len(feat_nm_aug)):
    ax = fig.add_subplot(15, 4, i+1)
    pl.hist(X_train[:,i])
    pl.title(feat_nm_aug[i])

# With categorical variables there was this discussion about variables that split
# the data set into two, or n-divisions I guess, is good for a tree algorithm.
# On the other hand isn't it possible that there are features that are non-descriptive
# most of the time but then if they are true they become very descriptive?
# Possible, but I guess they are still not very efficient discriminators because
# they apply to a small subset of data

# Also, many features are not efficiently split, like sexF = 1-sexM
# Check
z = X_train_df.sex_M + X_train_df.sex_F
any(z==0)
# Ok

# Note a digital variable that has n/2 0's and n/2 1's is maximally variable (max var)
# So screening for higher var is probably a good thing

# In problems where we have 2 outcomes and there is one much rarer than the other
# It might make sense to focus on features that are specifically good at getting
# that one right. Maybe ...

# Is there a concept of input variance vs. output variance?
# Meaning: if variance is a potentially good candidate for discrimination within
# the population then should it matter that also the ouput has good variance?

# It seems like: the higher the input variance is and the lower the conditional
# variance of the output, the better it is:
# high input variance gives us a good way to distinguish/split the data set, 
# low conditional output variance means that it has high predictive power.
# E.g. sex = M, F. If is turns out that M implies almost always yes and
# F implies almost always no, then you have a very good feature

# Let's check:
ixM = X_train_df.sex_M==1
len(y_train_d[X_train_df.sex_M==0])
X_tmp = X_train[X_train_df.iloc(X_train_df['sex_M']==1),:]
X_tmp = X_train[X_train_df['sex_M']==1,:]       # does not throw error, but does not do what I want
X_tmp = X_train_df[X_train_df['sex_M']==1,:]    # throws an error
# Damn!! this is soooo frustrating!!
X_tmp = X_train_df[X_train_df['sex_M']==1]    # damn, this worked
y_tmp = y_train_d[X_train_df['sex_M']==1]       # this did not work!
  
# Try with data version
y_tmp = y_train_d[X_train['sex_M']==1]       # this did not work either!
  
X_tmp = X_train_df[X_train_df['sex_M']==1]    # damn, this worked
yM_tmp = []
yF_tmp = []
for i in range(X_train_df.shape[0]):
    if X_train_df['sex_M'][i]==1:
        yM_tmp.append(y_train_d[i])
    else:
        yF_tmp.append(y_train_d[i])


import matplotlib.pyplot as pl
fig = pl.figure()
pl.hist(yM_tmp)
pl.title("Passing Males")

fig = pl.figure()
pl.hist(yF_tmp)
pl.title("Passing Females")

print float(sum(yM_tmp))/len(yM_tmp)
print float(sum(yF_tmp))/len(yF_tmp)

# Almost same percentage





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



# Porca figa ...
# Perconvertire df a numeri usa .as_matrix()!!!!




# Actual code


# Import libraries
import numpy as np
import pandas as pd

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns

# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
nO, nF = student_data.shape
col_nm = student_data.columns.tolist()
pss = student_data[col_nm[-1]]

isPass = np.array(pss=='yes')
nPass = sum(isPass==True)
nFail = sum(isPass==False)
grad = float(nPass) / float(nO)

n_students = nO
n_features = nF
n_passed = nPass
n_failed = nFail
grad_rate = grad*100
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

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

X_all_aug = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all_aug.columns), list(X_all_aug.columns))



# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

#print X_all_aug[0,0]
#print X_all_aug[0,1]
# X, y = X_all_aug.as_matrix(), y_all.as_matrix()
X, y = X_all_aug.as_matrix(), y_all
print type(X)
print X[0,0]
print X[0,1]
print y_all[0]


# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)


X_train, y_train = X[:num_train,:], y[:num_train]
X_test, y_test = X[num_train:,:], y[num_train:]

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
    return end - start

# TODO: Choose a model, import it and instantiate an object
from sklearn import svm
clf = svm.SVC()

# Fit model to training data
train_classifier(clf, X_train, y_train)  # note: using entire training set here
print clf  # you can inspect the learned model by printing it


# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    t_start = time.time()
    y_pred = clf.predict(features)
    t_end = time.time()
    dt = t_end - t_start
    print "Done!\nPrediction time (secs): {:.3f}".format(dt)
    return (f1_score(target.values, y_pred, pos_label='yes'), dt)

train_f1_score, dt = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)


# Predict on test data
f1_tmp, dt_tmp = predict_labels(clf, X_test, y_test)
print "F1 score for test set: {}".format(f1_tmp)


# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    dt_train = train_classifier(clf, X_train, y_train)
    f1_train, dt_train_prd = predict_labels(clf, X_train, y_train)
    f1_test, dt_test_prd = predict_labels(clf, X_test, y_test)
    print "Training time (secs): {:.3f}".format(dt_train)
    print "F1 score for training set: {}".format(f1_train)
    print "F1 score for test set: {}".format(f1_test)
    return (f1_train, f1_test, dt_train, dt_test_prd)

# TODO: Run the helper function above for desired subsets of training data
# Note: Keep the test set constant

X1, y1 = X_train[0:100,:], y_train[0:100] 
X2, y2 = X_train[0:200,:], y_train[0:200] 
X3, y3 = X_train, y_train # memory inefficient, but improving legibility 
f11_train, f11_test, dt1_train, dt1_test = train_predict(clf, X1, y1, X_test, y_test)
f12_train, f12_test, dt2_train, dt2_test = train_predict(clf, X2, y2, X_test, y_test)
f13_train, f13_test, dt3_train, dt3_test = train_predict(clf, X3, y3, X_test, y_test)

from pandas import DataFrame
df_SVM = DataFrame(columns={'train', 'test', 'train time', 'test time'})
df_SVM['train'] = [f11_train, f12_train, f13_train]
df_SVM['test'] = [f11_test, f12_test, f13_test]
df_SVM['train time'] = [dt1_train, dt2_train, dt3_train]
df_SVM['test time'] = [dt1_test, dt2_test, dt3_test]
df_SVM = df_SVM[['train', 'test', 'train time', 'test time']]


from IPython.display import display, HTML
print "SVM"
display(df_SVM)

# TODO: Train and predict using two other models
# Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')

f11_train, f11_test, dt1_train, dt1_test = train_predict(clf, X1, y1, X_test, y_test)
f12_train, f12_test, dt2_train, dt2_test = train_predict(clf, X2, y2, X_test, y_test)
f13_train, f13_test, dt3_train, dt3_test = train_predict(clf, X3, y3, X_test, y_test)

df_kNN = DataFrame(columns={'train', 'test', 'train time', 'test time'})
df_kNN['train'] = [f11_train, f12_train, f13_train]
df_kNN['test'] = [f11_test, f12_test, f13_test]
df_kNN['train time'] = [dt1_train, dt2_train, dt3_train]
df_kNN['test time'] = [dt1_test, dt2_test, dt3_test]
df_kNN = df_kNN[['train', 'test', 'train time', 'test time']]



# Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)

f11_train, f11_test, dt1_train, dt1_test = train_predict(clf, X1, y1, X_test, y_test)
f12_train, f12_test, dt2_train, dt2_test = train_predict(clf, X2, y2, X_test, y_test)
f13_train, f13_test, dt3_train, dt3_test = train_predict(clf, X3, y3, X_test, y_test)

df_DT = DataFrame(columns={'train', 'test', 'train time', 'test time'})
df_DT['train'] = [f11_train, f12_train, f13_train]
df_DT['test'] = [f11_test, f12_test, f13_test]
df_DT['train time'] = [dt1_train, dt2_train, dt3_train]
df_DT['test time'] = [dt1_test, dt2_test, dt3_test]
df_DT = df_DT[['train', 'test', 'train time', 'test time']]


print "SVM"
display(df_SVM)
print "\n"

print "kNN"
display(df_kNN)
print "\n"

print "DT"
display(df_DT)
print "\n"


# Gridsearch
# Choose scoring function f1_score
from sklearn.metrics import make_scorer, f1_score
sc_fnc = make_scorer(f1_score, greater_is_better=True, pos_label='yes')
prm = {'C':1}

from sklearn.grid_search import GridSearchCV
reg = GridSearchCV(clf,prm,scoring=sc_fnc,cv=10)
reg.fit(X_train,y_train)
reg.best_estimator_
reg.predict(CLIENT_FEATURES)















