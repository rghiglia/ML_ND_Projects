# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

## Make matplotlib show plots inline
#%matplotlib inline

# Create our client's feature set for which we will be predicting a selling price
CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]
type(CLIENT_FEATURES)   # list of lists?
type(CLIENT_FEATURES[0]) # correct
len(CLIENT_FEATURES)
len(CLIENT_FEATURES[0])

# Load Boston housing dataset
dir(datasets)
city_data = datasets.load_boston() # a dictionary
print city_data.items()
print city_data.keys()
city_data.feature_names

# Data consists of 506 observation of
# 13 features (city_data.feature_names)
# Data should be housing values in '000s?
# city_data.data has the values of the features
# city_data.targets should have the values of the homes (in '000s?)


# Step 1
data = np.array(city_data.data)
prc = city_data.target
feat_s = city_data.feature_names
# size: data.size gives all the elements
# size: data.shape gives the size
nR, nC = data.shape
total_houses = nR

prc.min()
prc.max()
print "Mean = ", np.mean(prc)
print "Median = ",np.median(prc)
print "Std = ",np.std(prc)

'''
    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population

'''


#d_cl = {f: CLIENT_FEATURES[i]} for i,f in enumerate(feat_s) not working
d_cl = {};
for i,f in enumerate(feat_s):
    # print i, f, CLIENT_FEATURES[0][i]
    d_cl[f] = CLIENT_FEATURES[0][i]

for f in ('TAX','RM','RAD'):
    print f, " = ", d_cl[f]


# Step 2
import random
#import math

# Example http://www.tutorialspoint.com/python/number_shuffle.htm
lst = [20, 16, 10, 5];
lst = [20, 16, 10, 5];
random.shuffle(lst)
print "Reshuffled list : ",  lst
# Ok, it doesn't return anything


def shuffle_split_data(X, y, pct = 0.30):
    ''' Shuffles and splits data'''
    nR, nC = X.shape
    ix = range(nR)
    random.shuffle(ix)
    iSplt = int(nR*pct)
    ix_trn= ix[:iSplt]
    ix_tst= ix[iSplt:]
    
    X_trn = X[ix_trn,:]
    y_trn = y[ix_trn]
    X_tst = X[ix_tst,:]
    y_tst = y[ix_tst]
    
#    X_trn, y_trn = X[:iSplt,:], y[:,iSplt]
#    X_tst, y_tst = X[iSplt:,:], y[iSplt]
    
    return X_trn, y_trn, X_tst, y_tst


#n_split = 3000
#X_train, X_test = X[:n_split], X[n_split:]
#y_train, y_test = y[:n_split], y[n_split:]

X_train, y_train, X_test, y_test = shuffle_split_data(data, prc)


# Step 3

from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression()
res = regr.fit(X_train,y_train)
type(res) #sklearn.linear_model.base.LinearRegression
y_hat_trn = regr.predict(X_train)
err_pred = y_hat_trn - y_train
s_err = sum(err_pred**2)
 
from sklearn import metrics
acc = metrics.accuracy_score(y_train,y_hat_trn)
acc = metrics.accuracy_score(list(y_train),list(y_hat_trn))
acc = metrics.accuracy_score(list(y_train[1:2]),list(y_hat_trn[1:2]))
print acc
metrics.precision_score(y_train,y_hat_trn)

metrics.accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))


y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
acc = metrics.accuracy_score(y_true,y_pred)
# Damn, why did the above not work?

y_pred = [0.5, 2, 1, 3]
y_true = [0, 1, 2, 3]
acc = metrics.accuracy_score(y_true,y_pred)
print acc
# This doesn't work, so maybe it's not accepting floats 
type(y_pred)
type(y_hat_trn)


from sklearn import metrics

def performance_metric(y_true, y_pred, mthd='MSE'):
    """ Calculates and returns the total error between true and predicted values
        based on a performance metric chosen by the student. """
        
    if mthd=='MSE':
        error = metrics.mean_squared_error(y_true, y_pred)
    elif mthd=='MAE':
        error = metrics.mean_absolute_error(y_true, y_pred)
    else:
        print "For regressions use MSE or MAE"
        error = None
    
    return error

print performance_metric(y_hat_trn,y_train)
y_hat_trn.shape
y_train.shape


# Step 4
from sklearn.tree import DecisionTreeRegressor
regrDTR = DecisionTreeRegressor()
prm = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
type(prm['max_depth'])
## Equivalent to this?
#prm = {'max_depth':tuple(np.array(range(10))+1)

from sklearn.metrics import make_scorer, mean_absolute_error
sc_fnc = make_scorer(mean_absolute_error)
from sklearn.grid_search import GridSearchCV
reg = GridSearchCV(regrDTR,prm,scoring=sc_fnc,cv=10)
reg.fit(X_train,y_train)
reg.best_estimator_
reg.predict(CLIENT_FEATURES)

reg2 = GridSearchCV(regrDTR,prm,scoring=sc_fnc,cv=10)
reg2.fit(data,prc)
reg2.best_estimator_
reg2.predict(CLIENT_FEATURES)

#reg = GridSearchCV(DecisionTreeRegressor(),prm,cv=10)
#reg = GridSearchCV(regr,prm,cv=10,scoring=fbeta_score)
#
#
#from sklearn.metrics import fbeta_score, make_scorer
#ftwo_scorer = make_scorer(fbeta_score, beta=2)
#ftwo_scorer
#from sklearn.grid_search import GridSearchCV
#from sklearn.svm import LinearSVC
#grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
#...                     scoring=ftwo_scorer)


# Interesting piece of code
sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)



def learning_curves(X_train, y_train, X_test, y_test):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing error rates for each model are then plotted. """
    
    print "Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . ."
    
    # Create the figure window
    fig = pl.figure(figsize=(10,8))

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    # Create four different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        for i, s in enumerate(sizes):
            
            # Setup a decision tree regressor so that it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth = depth)
            
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])

            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
            
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, regressor.predict(X_test))

        # Subplot the learning curve graph
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
        ax.plot(sizes, train_err, lw = 2, label = 'Training Error')
        ax.legend()
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize=18, y=1.03)
    fig.tight_layout()
    fig.show()



learning_curves(X_train, y_train, X_test, y_test)



def model_complexity(X_train, y_train, X_test, y_test):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    
    print "Creating a model complexity graph. . . "

    # We will vary the max_depth of a decision tree model from 1 to 14
    max_depth = np.arange(1, 14)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth = d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, test_err, lw=2, label = 'Testing Error')
    pl.plot(max_depth, train_err, lw=2, label = 'Training Error')
    pl.legend()
    pl.xlabel('Maximum Depth')
    pl.ylabel('Total Error')
    pl.show()


model_complexity(X_train, y_train, X_test, y_test)


# Hmmmm, I am concerned about the cross-validation giving me a best max_depth=1 ... ?


pl.hist(prc)
pl.title("Boston Prices")
pl.xlabel("000$")
pl.ylabel("Occurrances")
#fig = pl.gcf()
#plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')

nF = len(feat_s)

fig = pl.figure(figsize=(10,8))
for i,f in enumerate(feat_s):
    print i, f
    ax = fig.add_subplot(5, 3, i+1)
    ax.hist(data[:,i])
    ax.axvline(np.mean(data[:,i]), color='r', linestyle='--', linewidth=3)
    ax.axvline(CLIENT_FEATURES[0][i], color='y', linestyle='--', linewidth=3)
    ax.set_title(f)

CLIENT_FEATURES
d_cl

regressor = DecisionTreeRegressor(max_depth = 4)
regressor.fit(X_train, y_train)
regressor.predict(CLIENT_FEATURES)
regressor.fit(data, prc)
regressor.predict(CLIENT_FEATURES)
feat_s
