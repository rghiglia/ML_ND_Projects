# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:35:25 2016

@author: rghiglia
"""

# prj2_students_interv_20160322

#import numpy as np
#import pandas as pd
#from pandas import DataFrame
#
## Load data
#student_data = pd.read_csv(r"C:\Users\rghiglia\Documents\ML_ND\student_intervention\student-data.csv")
#type(student_data)  # dataframe
#
#nO, nF = student_data.shape
#col_nm = student_data.columns.tolist()
#pss = student_data[col_nm[-1]]
#
#isPass = np.array(pss=='yes')
#nPass = sum(isPass==True)
#nFail = sum(isPass==False)
#grad = float(nPass) / float(nO)
#
#feat_nm = col_nm[:-1]
#labl_nm = col_nm[-1]
#
## Some features assume only two values, e.g. yes/no, etc.
## Those can be converted into 0/1
#
## Some have multiple values and are called categorical
## The recommended way to convert them is to generate 0/1 for each value of the category
#
#
## Preprocess feature columns
#def preprocess_features(X):
#    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty
#
#    # Check each column
#    for col, col_data in X.iteritems():
#        # If data type is non-numeric, try to replace all yes/no values with 1/0
#        if col_data.dtype == object:
#            col_data = col_data.replace(['yes', 'no'], [1, 0])
#        # Note: This should change the data type for yes/no columns to int
#
#        # If still non-numeric, convert to one or more dummy variables
#        if col_data.dtype == object:
#            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'
#
#        outX = outX.join(col_data)  # collect column(s) in output dataframe
#
#    return outX
#
#
## Extract feature (X) and target (y) columns
#feature_cols = list(student_data.columns[:-1])  # all columns but last are features
#target_col = student_data.columns[-1]  # last column is the target/label
#print "Feature column(s):-\n{}".format(feature_cols)
#print "Target column: {}".format(target_col)
#
#X_all = student_data[feature_cols]  # feature values for all students
#y_all = student_data[target_col]  # corresponding targets/labels
#y_all_d = np.array(y_all.replace(['yes', 'no'], [1, 0])) # convert targets into 0s and 1s
#print "\nFeature values:-"
#print X_all.head()  # print the first 5 rows
#
## Friggin' amazing
#X_all_aug = preprocess_features(X_all) # careful: this converts the data into and array
#print "Processed feature columns ({}):-\n{}".format(len(X_all_aug.columns), list(X_all_aug.columns))
#nO_aug, nF_aug = X_all_aug.shape
#feat_nm_aug = list(X_all_aug.columns.values)
#
## First, decide how many training vs test samples you want
#num_all = student_data.shape[0]  # same as len(student_data)
#num_train = 300  # about 75% of the data
#num_test = num_all - num_train
#
## TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
## Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
#from sklearn.utils import shuffle
#X_all_aug_d, y_all, y_all_d = shuffle(X_all_aug, y_all, y_all_d, random_state=0)
## Careful: it converted X_all_aug into an 2D array
#type(X_all_aug_d)   # nd array
#type(y_all)         # nd array of yes, no
#type(y_all_d)       # nd array of 1s and 0s
#
#
## Split data set
#X_train_df = X_all_aug[:num_train,:]
## Error: unhashable type, so I cannot select a subset (slice) of a dataframe?
#X_train_df = X_all_aug.iloc[:num_train]
#
#X_train, y_train, y_train_d = X_all_aug_d[:num_train,:], y_all[:num_train], y_all_d[:num_train]
#X_test, y_test, y_test_d = X_all_aug_d[num_train:,:], y_all[num_train:], y_all_d[num_train:]
#print "Training set: {} samples".format(X_train.shape[0])
#print "Test set: {} samples".format(X_test.shape[0])
## Note: If you need a validation set, extract it from within training data
#
#
## Looking at features
#
## Almost all features are categorical, except age, and number of absences
#
#
#import matplotlib.pyplot as pl
#fig = pl.figure(figsize=(10,80))
#
## Histogram
#for i in range(len(feat_nm_aug)):
#    ax = fig.add_subplot(15, 4, i+1)
#    pl.hist(X_train[:,i])
#    pl.title(feat_nm_aug[i])
#
## With categorical variables there was this discussion about variables that split
## the data set into two, or n-divisions I guess, is good for a tree algorithm.
## On the other hand isn't it possible that there are features that are non-descriptive
## most of the time but then if they are true they become very descriptive?
## Possible, but I guess they are still not very efficient discriminators because
## they apply to a small subset of data
#
## Also, many features are not efficiently split, like sexF = 1-sexM
## Check
#z = X_train_df.sex_M + X_train_df.sex_F
#any(z==0)
## Ok
#
## Note a digital variable that has n/2 0's and n/2 1's is maximally variable (max var)
## So screening for higher var is probably a good thing
#
## In problems where we have 2 outcomes and there is one much rarer than the other
## It might make sense to focus on features that are specifically good at getting
## that one right. Maybe ...
#
## Is there a concept of input variance vs. output variance?
## Meaning: if variance is a potentially good candidate for discrimination within
## the population then should it matter that also the ouput has good variance?
#
## It seems like: the higher the input variance is and the lower the conditional
## variance of the output, the better it is:
## high input variance gives us a good way to distinguish/split the data set, 
## low conditional output variance means that it has high predictive power.
## E.g. sex = M, F. If is turns out that M implies almost always yes and
## F implies almost always no, then you have a very good feature
#
## Let's check:
#ixM = X_train_df.sex_M==1
#len(y_train_d[X_train_df.sex_M==0])
#X_tmp = X_train[X_train_df.iloc(X_train_df['sex_M']==1),:]
#X_tmp = X_train[X_train_df['sex_M']==1,:]       # does not throw error, but does not do what I want
#X_tmp = X_train_df[X_train_df['sex_M']==1,:]    # throws an error
## Damn!! this is soooo frustrating!!
#X_tmp = X_train_df[X_train_df['sex_M']==1]    # damn, this worked
#y_tmp = y_train_d[X_train_df['sex_M']==1]       # this did not work!
#  
## Try with data version
#y_tmp = y_train_d[X_train['sex_M']==1]       # this did not work either!
#  
#X_tmp = X_train_df[X_train_df['sex_M']==1]    # damn, this worked
#yM_tmp = []
#yF_tmp = []
#for i in range(X_train_df.shape[0]):
#    if X_train_df['sex_M'][i]==1:
#        yM_tmp.append(y_train_d[i])
#    else:
#        yF_tmp.append(y_train_d[i])
#
#
#import matplotlib.pyplot as pl
#fig = pl.figure()
#pl.hist(yM_tmp)
#pl.title("Passing Males")
#
#fig = pl.figure()
#pl.hist(yF_tmp)
#pl.title("Passing Females")
#
#print float(sum(yM_tmp))/len(yM_tmp)
#print float(sum(yF_tmp))/len(yF_tmp)
#
## Almost same percentage
#
#
#
#
#
## Train a model
#import time
#
#def train_classifier(clf, X_train, y_train):
#    print "Training {}...".format(clf.__class__.__name__)
#    start = time.time()
#    clf.fit(X_train, y_train)
#    end = time.time()
#    print "Done!\nTraining time (secs): {:.3f}".format(end - start)
#
## Predict on training set and compute F1 score
#from sklearn.metrics import f1_score, precision_score, recall_score
#
## Note: F1_score is presumably for classification problems
#
#def bool2YN(x):
#    y = []
#    for xi in x:
#        if xi==0:
#            y.append('no')
#        else:
#            y.append('yes')
#    return y
#
#def predict_labels(clf, features, target):
#    print "Predicting labels using {}...".format(clf.__class__.__name__)
#    start = time.time()
#    y_pred = clf.predict(features)
#    end = time.time()
#    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
#    if clf.__class__.__name__=='LinearRegression':
#        y_t = bool2YN(target)
#        y_p = bool2YN(np.maximum(np.minimum(np.round(y_pred),1),0))
#        p = precision_score(y_t, y_p, pos_label='yes')
#        r = recall_score(y_t, y_p, pos_label='yes')
#        print "Precision = ", p
#        print "Recall = ", r
#        return f1_score(y_t, y_p, pos_label='yes')
#    else:
##        y_t = bool2YN(target)
##        y_p = bool2YN(np.maximum(np.minimum(np.round(y_pred),1),0))
##        p = precision_score(y_t, y_p, pos_label='yes')
##        r = recall_score(y_t, y_p, pos_label='yes')
##        print "Precision = ", p
##        print "Recall = ", r
##        return f1_score(target.values, y_pred, pos_label='yes')
#        return f1_score(target, y_pred, pos_label='yes')
#
## TODO: Choose a model, import it and instantiate an object
#
## Linear Regression
#from sklearn import linear_model
#clf = linear_model.LinearRegression(fit_intercept=True, normalize=False)
#train_classifier(clf, X_train, y_train)
#train_f1_score_LR = predict_labels(clf, X_train, y_train)
#print "F1 score for training set: {}".format(train_f1_score_LR)
#
### Checks
##y_pred = clf.predict(X_train)
##y_p = bool2YN(np.maximum(np.minimum(np.round(y_pred),1),0))
##y_t = bool2YN(y_train)
##f1_score(y_t, y_p, pos_label='yes')
##p = precision_score(y_t, y_p, pos_label='yes')
##r = recall_score(y_t, y_p, pos_label='yes')
#
## Recall: ability to get positives
## Precison: not to screw up with false negatives
## F1: weighted average of precision and recall
#
#
## Support vector machine
#X_train, y_train = X_all[:num_train,:], y_all[:num_train]
#X_test, y_test = X_all[num_train:,:], y_all[num_train:]
#
#from sklearn import svm
#clf = svm.SVC()
#train_classifier(clf, X_train, y_train)
#train_f1_score_SVM = predict_labels(clf, X_train, y_train)
#print "F1 score for training set: {}".format(train_f1_score_SVM)
#
#
## Nearest Neighbor
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
#train_classifier(clf, X_train, y_train)
#train_f1_score_kNN = predict_labels(clf, X_train, y_train)
#print "F1 score for training set: {}".format(train_f1_score_kNN)
#
#
## Decision tree
#from sklearn.tree import DecisionTreeClassifier
## what's the decision tree regressor?
#clf = DecisionTreeClassifier(max_depth=5)
#train_classifier(clf, X_train, y_train)
#train_f1_score_DT = predict_labels(clf, X_train, y_train)
#print "F1 score for training set: {}".format(train_f1_score_DT)
## Ulla ... F1 = 1 ..., ok I think it had to do with max_depth
##f1_score(target, y_pred, pos_label='yes')
## Export tree
#from sklearn.externals.six import StringIO
#
## Damn! I still can't visualize the tree
#import sys
#sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\pydot-1.0.28')
#del sys.path[-1]
#import pydot
#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("iris.pdf")
#pydot.graph_from_dot_data(dot_data.getvalue())
#
#
#
## Porca figa ...
## Perconvertire df a numeri usa .as_matrix()!!!!




# Actual code


# Import libraries
import numpy as np
import pandas as pd

# Read student data
#student_data = pd.read_csv("student-data.csv")
## Load data
student_data = pd.read_csv(r"C:\Users\rghiglia\Documents\ML_ND\student_intervention\student-data.csv")
type(student_data)  # dataframe
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
y = pd.core.series.Series(y)    # force y back to being a Series


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
# Wow, it actually changes the object itself, doesn't make a local copy

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
prm = {'C':np.linspace(0.2,4,10)}

from sklearn.grid_search import GridSearchCV
clf = svm.SVC()
clf_gs = GridSearchCV(clf,prm,scoring=sc_fnc,cv=3)
clf_gs.fit(X_train,y_train)
clf_gs.best_estimator_
clf_gs.predict(X_test)

f1_score(y_test.values, clf_gs.predict(X_test), pos_label='yes')



# Compare classifications

clf = svm.SVC()




# Looking into features
x = X_all_aug[0]
X_all_aug[0] # gives me an error. This is weird
X_all_aug.ix[0] # first row
X_all_aug.ix[:,0] # first column

y_num = y_all.replace(['yes', 'no'],[1, 0])
ix_tmp_GP = X_all_aug[X_all_aug['school_GP']==1].index
y_GP = y_num[ix_tmp_GP]
ix_tmp_MS = X_all_aug[X_all_aug['school_MS']==1].index
y_MS = y_num[ix_tmp_MS]


import matplotlib.pyplot as pl
pl.figure()
pl.hist([y_GP, y_MS])
pl.title("Graduation")
pl.ylabel("Occurrances")

float(sum(y_GP))/len(y_GP)
float(sum(y_MS))/len(y_MS)

# How do I think about the inference?
# Suppose first I take one feature at a time

# If conditionally on that value the likelihood is larger, right?
# But I should somehow 'normalize' by the fact that not many graduate, right?

# How about variance of (graduation given school = MS) / variance (graduation)
from numpy import mean, std, var
var(y_GP)/var(y_num)
var(y_MS)/var(y_num)
# These are very close to 1. My intuition would be that they don't matter much

# Careful: you should rescale features (and outputs) to N(0,1) for non-binary inputs

def var_ratio(x,y):
    return var(x)/var(y)


# Loop over features
# For a quick check I will not normalize them. I will do it afterwards
# Well, actually only works for binary features (for now) or features that assume
# a value == 1. Needs to be a lot more robust
# Need to discretize them

vr = []
nO = []

for feat in X_all_aug.columns:
    ix = X_all_aug[X_all_aug[feat] == 1].index
    vr.append(var_ratio(y_num[ix],y_num))
    nO.append(len(ix))

from pandas import Series
vr2 = Series(vr, index = X_all_aug.columns, name='var_ratio')
nO2 = Series(nO, index = X_all_aug.columns, name='obs')
#Z = vr2.to_frame('var_ratio')
#Z = Z.join(nO2)
Z = DataFrame(index=vr2.index)
Z = Z.join(vr2)
Z = Z.join(nO2)


pl.bar(np.arange(len(vr)),vr)

ixs = Z.sort('var_ratio').index

ind = np.arange(Z.shape[0])
pl.figure(figsize=(9, 4))
pl.bar(ind,vr2[ixs])
pl.xticks(ind, ixs, rotation=90)

# It really should be the entropy ratio 




# Dev entropy meawsures

def hist_discr(z):
    # Assumes 'z' is a discrete variable
    z = Series(z)
    c_lev = set(z)
    nC = len(c_lev)
    n = []
    Ix = []
    for i in c_lev:
        z_tmp = z[z==i];
        n.append(len(z_tmp))
        Ix.append(z_tmp.index)
    n = Series(n,index=c_lev)
    return (n, Ix)

def prb_discr(z):
    n, Ix = hist_discr(z)
    return n/sum(n)

def entropy(z):
    p = prb_discr(z)
    return -sum(p* np.log2(p))
    


z = Series([0, 0, 1, 1, 0, 2, 1])
len(z[z==z[0]])

z1, Ix = hist_discr(z)
z2 = prb_discr(z)
z3 = entropy(z) # can be larger than 1 because you have 3 possible values
z3 = entropy([0, 0, 0, 1, 1, 1])
z3 = entropy([0, 0, 0, 0, 0, 0])
# Ok


# Working towards entropy gain
# If I have an attribute Ai assuming values \in {a_i1, ..., a_ij, ..., a_im}
# then you have n obs
# get the subsets of S, e.g. S1: Ai1 = {s \in S: s_k = A_i1}

# For each subset you calculate the entropy, right?

# Example: attribute school_GP
feat = 'school_GP'
z = X_all_aug[feat]
tmp, IxA = hist_discr(z)
ei = []
w = []
for ix in IxA:
    print ix
    w.append(float(len(ix))/len(y_num))
    ei.append(w[-1] * entropy(y_num[ix]))
    print w[-1]
    print ei[-1]
#e_gain = entropy(z) - sum(ei)
e_gain = entropy(y_num) - sum(ei)
# hmmm ... not sure the gain should be negative
# Ok, I think the mistake was the calculation of the entropy on the input variable ...

e_gain = []
for feat in X_all_aug.columns:
    z = X_all_aug[feat]
    tmp, IxA = hist_discr(z)
    ei = []
    w = []
    for ix in IxA:
        w.append(float(len(ix))/len(y_num))
        ei.append(w[-1] * entropy(y_num[ix]))
    e_gain.append(entropy(y_num) - sum(ei))
    print feat, ": ", e_gain[-1]
e_gain = Series(e_gain, index=X_all_aug.columns)
e_gain.sort()
ixs = e_gain.index

ind = np.arange(len(e_gain))
pl.figure(figsize=(9, 4))
pl.bar(ind,e_gain)
pl.xticks(ind, ixs, rotation=90)


# Kinda ok. First question is negative gain, there seems to be something fishy
# You should probably try to extract importance from the algo and compare it to
# what you have
# then you should try to simplify the data set

# How do I check entropy and gain? Maybe I need to reproduce the PlayTennis

# Because suppose you have an unconditional 
# Ok, I think I found the problem

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
z_tmp = Series(clf.feature_importances_, X_all_aug.columns)

z_tmp.sort()
ixs = z_tmp.index

ind = np.arange(len(e_gain))
pl.figure(figsize=(9, 4))
pl.bar(ind,e_gain)
pl.xticks(ind, ixs, rotation=90)

df = DataFrame({'e_gain': e_gain, 'import': z_tmp})
df = df.sort('e_gain', ascending=False)

ind = np.arange(df.shape[0])
pl.figure(figsize=(9, 4))
pl.bar(ind,df.values[:,0],df.values[:,1])
pl.xticks(ind, df.index, rotation=90)
# Not great plot

# Anyways, somewhat ok but result a bit different

# Should I look at the decision boundary of the SVM?

# How about redundancy analysis?
# Should I look into mutual information?

# I can use the above for relevance analysis


# I can try Hamming distance
# I want something tyhat is symmetric but goes from -1 to + 1
# Actually will use 0 to 1

def hamm_entropy(z1,z2):
#    z12 = sum(z1==z2) / len(z1)
#    z12n = sum(z1!=z2) / len(z1)
#    return min([z12, z12n])
    h = abs(z2 - z1)
    p = sum(h / len(z1))
    if p==0:
        d = 0
    else:
        d = max(-p*np.log2(p),0)
    #d(math.isnan(d)) = 1
    return d

z1 = X_all_aug.ix[:,0]
z2 = X_all_aug.ix[:,1]
h = abs(z2 - z1)
p = h / len(z1)
hamm_entropy(z1,z2)
hamm_entropy(X_all_aug.ix[:,0],X_all_aug.ix[:,3])

def Hamm_entropy(X):
    if type(X)==DataFrame:
        X = X.values
    nC = X.shape[1]
    H = np.zeros([nC, nC])
    for i in range(nC):
        for j in range(i+1,nC):
            H[i,j] = hamm_entropy(X[:,i],X[:,j])
            H[j,i] = H[i,j]
    return H

H = Hamm_entropy(X_all_aug)
type(X_all_aug)==DataFrame


pl.figure()
pl.scatter(X_all_aug.ix[:,0],X_all_aug.ix[:,1])
pl.scatter(X_all_aug.ix[:,0],X_all_aug.ix[:,2])
pl.scatter(X_all_aug.ix[:,0],X_all_aug.ix[:,3])
pl.scatter(X_all_aug.ix[:,0],X_all_aug.ix[:,4])


# Ok, you need to digitize

# Get the cdf and dividee by n, extract the inverse points



# A measure of vicinity and density?

# A bit of an issue is the mixure of digital and analog variables

# Use hamm_entropy for now

x_test = X_test[0,:]


hamm_entropy(X_all_aug.ix[0,:],x_test)
hamm_entropy(X_all_aug.ix[1,:],x_test)
hamm_entropy(X_all_aug.ix[2,:],x_test)


nO = X_train.shape[0]
d1 = np.zeros([nO, 1])

for i in range(nO):
    d1[i] = hamm_entropy(X_all_aug.ix[i,:],x_test)

pl.figure()
pl.hist(d1) # wtf! I got an error: Ok, pl.hist doesn't handle NaN's

mean(d1)

# Let's compare it to an in-sample point:

for i in range(nO):
    d1[i] = hamm_entropy(X_train[i,:],x_test)
pl.figure()
pl.hist(d1) # wtf! I got an error: Ok, pl.hist doesn't handle NaN's

mean(d1)

# This could give you a sense of relative confidence

# Then you should check the relative closeness to only n neighbors
# Answering the question of how close are you to the say 3 nearest neighbors

def hamm_entropy_Xy(X,y):
    if type(X)==DataFrame: X = X.values
    nO = X.shape[0]
    d1 = np.zeros([nO,1])
    for i in range(nO):
        d1[i] = hamm_entropy(X[i,:],y)
    return d1[:,0]

def hamm_entropy_kNN(X,y,kNN=3):
    d1 = hamm_entropy_Xy(X,y)
    d1 = Series(d1)
    d1.sort(ascending=True)
    ixs = d1.index
    d1_kNN = d1[0:kNN]
    ixs_kNN = ixs[0:kNN]
    return (np.array(d1_kNN), ixs_kNN)

pl.figure()
pl.hist(d1)

d1_kNN, ixs_kNN = hamm_entropy_kNN(X_train,x_test, kNN=10)
pl.figure()
pl.hist(d1_kNN)
Z = X_train[ixs_kNN,:]
# WTF is going on?
hamm_entropy(X_train[ixs_kNN[0],:],x_test)

z1_org = X_train[ixs_kNN[0],:]
z2_org = x_test
z1 = z1_org - mean(z1_org)
z1[z1>0] = 1
z1[z1<=0] = 0
h = abs(z2 - z1)
p = sum(h / len(z1))
# Ahh, yeah, it will break for non-digital entries


d1_kNN = hamm_entropy_kNN(X_train, X_train[0,:], kNN=10)
pl.figure()
pl.hist(d1_kNN)

# Still not right
# You need to transform the original data! By column, not by row ... you cannot do it with the hamm function


Z_all_aug = X_all_aug
Z_all_aug = np.sign(Z_all_aug - mean(Z_all_aug))
Z_all_aug[Z_all_aug==-1] = 0