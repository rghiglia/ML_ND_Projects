# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:35:25 2016

@author: rghiglia
"""

# prj2_students_interv_20160322

# Import libraries
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


# -----------------------------------------------------------------------------
# Read/load the data
# -----------------------------------------------------------------------------

student_data = pd.read_csv(r"C:\Users\rghiglia\Documents\ML_ND\student_intervention\student-data.csv")
type(student_data)  # dataframe
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns



# -----------------------------------------------------------------------------
# Pre-process the data
# -----------------------------------------------------------------------------

# This will be a generic problem: given the data convert it into useable format
# Data (input and/or output) could be all numeric or a mixture
# In general it would seems best to convert the data into DataFrames with all numerical values
# The you can think of converting it into multiple columns for categorical variables

data = student_data             # make sure it's a data frame

# Write a routine that will convert categorical variables in numbers
# The following routine is starting to do (a lot) more, but for now just
# use data_mlnd. Since I am pre-processing the data before splitting in features
# and target it will guarantee everything is numeric
import sys
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND')
from rg_toolbox_data import preproc_data

data_num, data_bol, data_bin, data_mlnd = preproc_data(data)

features, target = data_mlnd.ix[:,:-1], data_mlnd.ix[:,-1]    # features = df, target = Series (could still have non-numerical values)
type(features)
type(target)
y_tgt = target


# -----------------------------------------------------------------------------
# Initial look at the data
# -----------------------------------------------------------------------------

# Original data set
nO, nF = data.shape
nF_mlnd = data_mlnd.shape[1]
all_nm = data.columns.tolist()
feat_nm = all_nm[:-1]
tgt_nm = all_nm[-1]
feat_mlnd_nm = data_mlnd.columns.tolist(); feat_mlnd_nm.pop()     # drop last one

pss = data_mlnd['passed']
nPass = sum(pss==1)
nFail = sum(pss==0)
grad = float(nPass) / float(nO)
print "Pass = ", nPass, ", Fail = ", nFail, ", Graduation rate = %1.1f %%" % (100*grad)

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


# -----------------------------------------------------------------------------
# Prepare and split data set
# -----------------------------------------------------------------------------

# Following is really to reproduce the file for the class
# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_org = student_data[feature_cols]  # feature values for all students
y_org = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_org.head()  # print the first 5 rows

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

X_org_aug = preprocess_features(X_org)
print "Processed feature columns ({}):-\n{}".format(len(X_org_aug.columns), list(X_org_aug.columns))

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300                  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
from sklearn.utils import shuffle
ixs = shuffle(X_org_aug.index, random_state=0)
X_tmp_aug, y_tmp, y_tpp = X_org_aug.reindex(ixs), y_org.reindex(ixs), y_tgt.reindex(ixs)
# Check
df = DataFrame({'yn': y_tmp, 'bol': y_tpp}, index=X_tmp_aug.index)

X_train, y_train, y_train_bol = X_tmp_aug.ix[ixs[:num_train],:], y_tmp[ixs[:num_train]], y_tpp[ixs[:num_train]]
X_test, y_test, y_test_bol = X_tmp_aug.ix[ixs[num_train:],:], y_tmp[ixs[num_train:]], y_tpp[ixs[num_train:]]
# Check
df = DataFrame({'yn': y_train, 'bol': y_train_bol}, index=X_train.index)


print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data




# -----------------------------------------------------------------------------
# Closer look at the data
# -----------------------------------------------------------------------------


# Some stats about Entropy gain, importance, Hamtropy, etc.

# Entropy gain
from rg_toolbox_math import entropy_gain
#eg = Series([entropy_gain(data_mlnd[feat],y) for feat in feat_mlnd_nm], index=feat_mlnd_nm)
eg = Series([entropy_gain(X_train[feat],y_train_bol) for feat in X_train.columns], index=feat_mlnd_nm)
#n, Ix = hist_discr(X_train['school_GP'])
#entropy_gain(X_train['school_GP'],y_train_bol)
egs = eg.order(ascending=False)
ixs_eg = egs.index

import matplotlib.pyplot as pl
ind = np.arange(len(egs))
fig = pl.figure(1, figsize=(9, 4))
ax = fig.add_subplot(111)
ax.bar(ind,egs)
pl.xticks(ind, ixs_eg, rotation=90)
ax.set_title('Entropy Gain')

# Using a tree for feature importance
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X_train, y_train)
z = Series(clf.feature_importances_, X_train.columns)
zs = z.order(ascending=False)
ixsDT = zs.index

df_eg = DataFrame({'e_gain': egs}, index=ixs_eg)
df_DT = DataFrame({'DT': zs}, index=ixsDT)
df = df_eg.join(df_DT)


ind = np.arange(df.shape[0])
fig = pl.figure(figsize=(9, 4))
ax = fig.add_subplot(111)
ax.bar(ind,df.e_gain, width=0.25)
ax.bar(ind+0.5,df.DT, width=0.25, color='g')
pl.xticks(ind, ixs_eg, rotation=90)

# Some agreement, although some are interestingly different
# Following results with criterion='Gini'
# Interestingly if if max_depth = 1, it picks the single most relevant and it's failures
# With max_depth = 2, we have failures, absences, and schoolsup
# With max_depth = 3, we also see freetime, health, studytime
# With max_depth = 10, it starts being a bit more similar to entropy gain




# -----------------------------------------------------------------------------
# Train a model
# -----------------------------------------------------------------------------
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

X1, y1 = X_train.ix[ixs[0:100],:], y_train[ixs[0:100]]
X2, y2 = X_train.ix[ixs[0:200],:], y_train[ixs[0:200]]
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
print "kNN"
display(df_kNN)



# Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3)

f11_train, f11_test, dt1_train, dt1_test = train_predict(clf, X1, y1, X_test, y_test)
f12_train, f12_test, dt2_train, dt2_test = train_predict(clf, X2, y2, X_test, y_test)
f13_train, f13_test, dt3_train, dt3_test = train_predict(clf, X3, y3, X_test, y_test)

df_DT = DataFrame(columns={'train', 'test', 'train time', 'test time'})
df_DT['train'] = [f11_train, f12_train, f13_train]
df_DT['test'] = [f11_test, f12_test, f13_test]
df_DT['train time'] = [dt1_train, dt2_train, dt3_train]
df_DT['test time'] = [dt1_test, dt2_test, dt3_test]
df_DT = df_DT[['train', 'test', 'train time', 'test time']]
print "DT"
display(df_DT)


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



## Compare classifications
#
#clf = svm.SVC()
#
#
#
#
## Looking into features
#x = X_all_aug[0]
#X_all_aug[0] # gives me an error. This is weird
#X_all_aug.ix[0] # first row
#X_all_aug.ix[:,0] # first column
#
#y_num = y_all.replace(['yes', 'no'],[1, 0])
#ix_tmp_GP = X_all_aug[X_all_aug['school_GP']==1].index
#y_GP = y_num[ix_tmp_GP]
#ix_tmp_MS = X_all_aug[X_all_aug['school_MS']==1].index
#y_MS = y_num[ix_tmp_MS]
#
#
#import matplotlib.pyplot as pl
#pl.figure()
#pl.hist([y_GP, y_MS])
#pl.title("Graduation")
#pl.ylabel("Occurrances")
#
#float(sum(y_GP))/len(y_GP)
#float(sum(y_MS))/len(y_MS)
#
## How do I think about the inference?
## Suppose first I take one feature at a time
#
## If conditionally on that value the likelihood is larger, right?
## But I should somehow 'normalize' by the fact that not many graduate, right?
#
## How about variance of (graduation given school = MS) / variance (graduation)
#from numpy import mean, std, var
#var(y_GP)/var(y_num)
#var(y_MS)/var(y_num)
## These are very close to 1. My intuition would be that they don't matter much
#
## Careful: you should rescale features (and outputs) to N(0,1) for non-binary inputs
#
#def var_ratio(x,y):
#    return var(x)/var(y)
#
#
## Loop over features
## For a quick check I will not normalize them. I will do it afterwards
## Well, actually only works for binary features (for now) or features that assume
## a value == 1. Needs to be a lot more robust
## Need to discretize them
#
#vr = []
#nO = []
#
#for feat in X_all_aug.columns:
#    ix = X_all_aug[X_all_aug[feat] == 1].index
#    vr.append(var_ratio(y_num[ix],y_num))
#    nO.append(len(ix))
#
#from pandas import Series
#vr2 = Series(vr, index = X_all_aug.columns, name='var_ratio')
#nO2 = Series(nO, index = X_all_aug.columns, name='obs')
##Z = vr2.to_frame('var_ratio')
##Z = Z.join(nO2)
#Z = DataFrame(index=vr2.index)
#Z = Z.join(vr2)
#Z = Z.join(nO2)
#
#
#pl.bar(np.arange(len(vr)),vr)
#
#ixs = Z.sort('var_ratio').index
#
#ind = np.arange(Z.shape[0])
#pl.figure(figsize=(9, 4))
#pl.bar(ind,vr2[ixs])
#pl.xticks(ind, ixs, rotation=90)
#
## It really should be the entropy ratio 
#
#
### Note:
##hist, bin_edges = np.histogram(a)
##hist, bin_edges = np.histogram(a, density=True)
## Same as Matlab [n, b] = hist(a)
#
#
## Dev entropy meawsures
#
#def hist_discr(z):
#    # Assumes 'z' is a discrete variable
#    z = Series(z)
#    c_lev = set(z)
#    nC = len(c_lev)
#    n = []
#    Ix = []
#    for i in c_lev:
#        z_tmp = z[z==i];
#        n.append(len(z_tmp))
#        Ix.append(z_tmp.index)
#    n = Series(n,index=c_lev)
#    return (n, Ix)
#
#def prb_discr(z):
#    n, Ix = hist_discr(z)
#    return n/sum(n)
#
#def entropy(z):
#    p = prb_discr(z)
#    return -sum(p* np.log2(p))
#    
#
#
#z = Series([0, 0, 1, 1, 0, 2, 1])
#len(z[z==z[0]])
#
#z1, Ix = hist_discr(z)
#z2 = prb_discr(z)
#z3 = entropy(z) # can be larger than 1 because you have 3 possible values
#z3 = entropy([0, 0, 0, 1, 1, 1])
#z3 = entropy([0, 0, 0, 0, 0, 0])
## Ok
#
#
## Working towards entropy gain
## If I have an attribute Ai assuming values \in {a_i1, ..., a_ij, ..., a_im}
## then you have n obs
## get the subsets of S, e.g. S1: Ai1 = {s \in S: s_k = A_i1}
#
## For each subset you calculate the entropy, right?
#
## Example: attribute school_GP
#feat = 'school_GP'
#z = X_all_aug[feat]
#tmp, IxA = hist_discr(z)
#ei = []
#w = []
#for ix in IxA:
#    print ix
#    w.append(float(len(ix))/len(y_num))
#    ei.append(w[-1] * entropy(y_num[ix]))
#    print w[-1]
#    print ei[-1]
##e_gain = entropy(z) - sum(ei)
#e_gain = entropy(y_num) - sum(ei)
## hmmm ... not sure the gain should be negative
## Ok, I think the mistake was the calculation of the entropy on the input variable ...
#
#e_gain = []
#for feat in X_all_aug.columns:
#    z = X_all_aug[feat]
#    tmp, IxA = hist_discr(z)
#    ei = []
#    w = []
#    for ix in IxA:
#        w.append(float(len(ix))/len(y_num))
#        ei.append(w[-1] * entropy(y_num[ix]))
#    e_gain.append(entropy(y_num) - sum(ei))
#    print feat, ": ", e_gain[-1]
#e_gain = Series(e_gain, index=X_all_aug.columns)
#e_gain.sort()
#ixs = e_gain.index
#
#ind = np.arange(len(e_gain))
#pl.figure(figsize=(9, 4))
#pl.bar(ind,e_gain)
#pl.xticks(ind, ixs, rotation=90)
#
#
## Kinda ok. First question is negative gain, there seems to be something fishy
## You should probably try to extract importance from the algo and compare it to
## what you have
## then you should try to simplify the data set
#
## How do I check entropy and gain? Maybe I need to reproduce the PlayTennis
#
## Because suppose you have an unconditional 
## Ok, I think I found the problem
#
#clf = DecisionTreeClassifier(max_depth=5)
#clf.fit(X_train, y_train)
#z_tmp = Series(clf.feature_importances_, X_all_aug.columns)
#
#z_tmp.sort()
#ixs = z_tmp.index
#
#ind = np.arange(len(e_gain))
#pl.figure(figsize=(9, 4))
#pl.bar(ind,e_gain)
#pl.xticks(ind, ixs, rotation=90)
#
#df = DataFrame({'e_gain': e_gain, 'import': z_tmp})
#df = df.sort('e_gain', ascending=False)
#
#ind = np.arange(df.shape[0])
#pl.figure(figsize=(9, 4))
#pl.bar(ind,df.values[:,0],df.values[:,1])
#pl.xticks(ind, df.index, rotation=90)
## Not great plot
#
## Anyways, somewhat ok but result a bit different
#
## Should I look at the decision boundary of the SVM?
#
## How about redundancy analysis?
## Should I look into mutual information?
#
## I can use the above for relevance analysis
#
#
## I can try Hamming distance
## I want something tyhat is symmetric but goes from -1 to + 1
## Actually will use 0 to 1
#
#def hamm_entropy(z1,z2):
##    z12 = sum(z1==z2) / len(z1)
##    z12n = sum(z1!=z2) / len(z1)
##    return min([z12, z12n])
#    h = abs(z2 - z1)
#    p = sum(h / len(z1))
#    if p==0:
#        d = 0
#    else:
#        d = max(-p*np.log2(p),0)
#    #d(math.isnan(d)) = 1
#    return d
#
#z1 = X_all_aug.ix[:,0]
#z2 = X_all_aug.ix[:,1]
#h = abs(z2 - z1)
#p = h / len(z1)
#hamm_entropy(z1,z2)
#hamm_entropy(X_all_aug.ix[:,0],X_all_aug.ix[:,3])
#
#def Hamm_entropy(X):
#    if type(X)==DataFrame:
#        X = X.values
#    nC = X.shape[1]
#    H = np.zeros([nC, nC])
#    for i in range(nC):
#        for j in range(i+1,nC):
#            H[i,j] = hamm_entropy(X[:,i],X[:,j])
#            H[j,i] = H[i,j]
#    return H
#
#H = Hamm_entropy(X_all_aug)
#type(X_all_aug)==DataFrame
#
#
#pl.figure()
#pl.scatter(X_all_aug.ix[:,0],X_all_aug.ix[:,1])
#pl.scatter(X_all_aug.ix[:,0],X_all_aug.ix[:,2])
#pl.scatter(X_all_aug.ix[:,0],X_all_aug.ix[:,3])
#pl.scatter(X_all_aug.ix[:,0],X_all_aug.ix[:,4])
#
#
## Ok, you need to digitize
#
## Get the cdf and dividee by n, extract the inverse points
#
#
#
## A measure of vicinity and density?
#
## A bit of an issue is the mixure of digital and analog variables
#
## Use hamm_entropy for now
#
#x_test = X_test[0,:]
#
#
#hamm_entropy(X_all_aug.ix[0,:],x_test)
#hamm_entropy(X_all_aug.ix[1,:],x_test)
#hamm_entropy(X_all_aug.ix[2,:],x_test)
#
#
#nO = X_train.shape[0]
#d1 = np.zeros([nO, 1])
#
#for i in range(nO):
#    d1[i] = hamm_entropy(X_all_aug.ix[i,:],x_test)
#
#pl.figure()
#pl.hist(d1) # wtf! I got an error: Ok, pl.hist doesn't handle NaN's
#
#mean(d1)
#
## Let's compare it to an in-sample point:
#
#for i in range(nO):
#    d1[i] = hamm_entropy(X_train[i,:],x_test)
#pl.figure()
#pl.hist(d1) # wtf! I got an error: Ok, pl.hist doesn't handle NaN's
#
#mean(d1)
#
## This could give you a sense of relative confidence
#
## Then you should check the relative closeness to only n neighbors
## Answering the question of how close are you to the say 3 nearest neighbors
#
#def hamm_entropy_Xy(X,y):
#    if type(X)==DataFrame: X = X.values
#    nO = X.shape[0]
#    d1 = np.zeros([nO,1])
#    for i in range(nO):
#        d1[i] = hamm_entropy(X[i,:],y)
#    return d1[:,0]
#
#def hamm_entropy_kNN(X,y,kNN=3):
#    d1 = hamm_entropy_Xy(X,y)
#    d1 = Series(d1)
#    d1.sort(ascending=True)
#    ixs = d1.index
#    d1_kNN = d1[0:kNN]
#    ixs_kNN = ixs[0:kNN]
#    return (np.array(d1_kNN), ixs_kNN)
#
#pl.figure()
#pl.hist(d1)
#
#d1_kNN, ixs_kNN = hamm_entropy_kNN(X_train,x_test, kNN=10)
#pl.figure()
#pl.hist(d1_kNN)
#Z = X_train[ixs_kNN,:]
## WTF is going on?
#hamm_entropy(X_train[ixs_kNN[0],:],x_test)
#
#z1_org = X_train[ixs_kNN[0],:]
#z2_org = x_test
#z1 = z1_org - mean(z1_org)
#z1[z1>0] = 1
#z1[z1<=0] = 0
#h = abs(z2 - z1)
#p = sum(h / len(z1))
## Ahh, yeah, it will break for non-digital entries
#
#
#d1_kNN = hamm_entropy_kNN(X_train, X_train[0,:], kNN=10)
#pl.figure()
#pl.hist(d1_kNN)
#
## Still not right
## You need to transform the original data! By column, not by row ... you cannot do it with the hamm function
#
#
#Z_all_aug = X_all_aug
#Z_all_aug = np.sign(Z_all_aug - mean(Z_all_aug))
#Z_all_aug[Z_all_aug==-1] = 0
#
## Normalization wrong!
#
#
#Z = Z_all_aug.values
#Z, y = shuffle(Z, y, random_state=0)
#y = pd.core.series.Series(y)    # force y back to being a Series
#
#Z_train, z_train = Z[:num_train,:], y[:num_train]
#Z_test, z_test = Z[num_train:,:], y[num_train:]
#
#d1_kNN, ixs_kNN = hamm_entropy_kNN(Z_train, Z_train[0,:], kNN=10)
#pl.figure()
#pl.hist(d1_kNN)
## Now it's much better
#
#d1_kNN, ixs_kNN = hamm_entropy_kNN(Z_train, Z_test[1,:], kNN=10)
#pl.figure()
#pl.hist(d1_kNN)
#
## Now I'd like to compare each test point with the average of the training points
#
## Get the mean kNN Hamtropy distance for each training point
#n_trn = Z_train.shape[0]
#n_tst = Z_test.shape[0]
##d1 = 0*np.array(range(nT))
#d1_train = np.zeros([n_trn,1])
#for i in range(n_trn):
#    d1_kNN, ixs  = hamm_entropy_kNN(Z_train,Z_train[i,:])
#    d1_train[i] = mean(d1_kNN[1:]) # remove the point itself
#    print mean(d1_kNN)
#
#d1_test = np.zeros([n_tst,1])
#for i in range(n_tst):
#    d1_kNN, ixs  = hamm_entropy_kNN(Z_train,Z_test[i,:])
#    d1_test[i] = mean(d1_kNN)
#    print mean(d1_kNN)
#
#
#Z_tmp = Z_train[[0, 135],:]
#
#pl.figure()
#pl.hist(d1_train)
#
#
#pl.figure()
#pl.hist(d1_test)
#
#print mean(d1_train)
#print mean(d1_test)
#
#
## Compare distance between prediction and actual
#y_pred = clf.predict(X_train)
#y_pred[y_pred=='yes'] = 1
#y_pred[y_pred=='no'] = 0
#y_pred = Series(y_pred)
#X_cmp = DataFrame({'y_train': y_train.replace(['yes', 'no'],[1, 0]), 'y_pred': y_pred})
#
## Ok, so from here you could provide a confidence measure by how many misclassifications
## you have in the neighborhood
#
## In continuous domain you could use confidence: 1 - distance(closest misclass)/distance(furthest point)
#
## I need a list of neighbors and their distance (should be ok with above functions but also their classification result)
#
#def hamm_entropy_xXy(x,X,y,y_pred):
#    # x: observation under consideration
#    # X: potential neighbors
#    # y: classification of neighbors in X
#    if type(X)==DataFrame: X = X.values
#    nO = X.shape[0]
#    d1 = np.zeros([nO,1])
#    for i in range(nO):
#        d1[i] = hamm_entropy(X[i,:],x)
#    d1 = d1[:,0]
#    y = Series(y).replace(['yes', 'no'],[1, 0])
#    y_pred = Series(y_pred).replace(['yes', 'no'],[1, 0])
#    w = 1.0 / d1
#    w = w / sum(w)
#    y_tt = y==y_pred
#    df = DataFrame({'d1': d1, 'act': y, 'pred': y_pred, 
#    'w': w, 'cls': y_tt.astype(int)})
#    df = df.sort('d1')
#    df = df[['d1','w','act','pred', 'cls']]
#    return df
#
#df = hamm_entropy_xXy(Z_test[0,:], Z_train, y_train, y_pred)
#conf = sum(df.w * df.cls)
#
## Ok, now you can do it for all testing points
#
#
#conf_test = np.zeros([n_tst,1])
#for i in range(n_tst):
#    df = hamm_entropy_xXy(Z_test[i,:], Z_train, y_train, y_pred)
#    conf_test[i] = sum(df.w * df.cls)
#
#
#
## 4/26/2016
#
## I wanted to see if I could reduce the number of features and get better results
#
## I think I really need a clean file here ...
#
#
#
