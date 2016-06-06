# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:53:18 2016

@author: rghiglia
"""

# customer_segment

# As part of this project I also want to review and reorg my pre-process


# -----------------------------------------------------------------------------
# Path, Packages, Etc.
# -----------------------------------------------------------------------------
import sys; sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\customer_segment')
#from rg_toolbox_data import cut_to_df, preproc_data, cat2num

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from matplotlib import pyplot as pl
from IPython.display import display # Allows the use of display() for DataFrames



# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
dnm = r'C:\Users\rghiglia\Documents\ML_ND\customer_segment'
fnm = 'customers.csv'
fnmL= dnm + '\\' + fnm

# If data is split into training and testing it might be a good idea to join it
## Joining training data and test data
#df_trn = pd.read_csv(r'C:\Users\rghiglia\Documents\ML_ND\Kaggle\Titanic\train.csv', header=0)
#df_tst = pd.read_csv(r'C:\Users\rghiglia\Documents\ML_ND\Kaggle\Titanic\test.csv', header=0)

# Load the wholesale customers dataset
try:
    data = pd.read_csv(fnmL)
    # Customized for this problem:
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# Display a description of the dataset
data.head()
data.describe()


## Q1
#Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.
#What kind of establishment (customer) could each of the three samples you've chosen represent?
#Hint: Examples of establishments include places like markets, cafes, and retailers, among many others. Avoid using names for establishments, such as saying "McDonalds" when describing a sample customer as a restaurant.

# I did most of it in notebook


#import sklearn.cluster



# Actual code
# Import libraries necessary for this project
#import numpy as np
#import pandas as pd
import renders as rs
#from IPython.display import display # Allows the use of display() for DataFrames

## Show matplotlib plots inline (nicely formatted in the notebook)
#%matplotlib inline

## Load the wholesale customers dataset
#try:
#    data = pd.read_csv("customers.csv")
#    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
#    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
#except:
#    print "Dataset could not be loaded. Is the dataset missing?"
#
## Display a description of the dataset
#display(data.describe())

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [95, 181, 85]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Table 0.1: Chosen samples of wholesale customers dataset:"
display(samples)

import seaborn as sns

# Look at percentile ranks
pcts = 100. * data.rank(axis=0, pct=True).iloc[indices].round(decimals=3)
print "\nTable 0.2: Chosen samples percentile ranks"
display(pcts)

# Visualize percentiles with heatmap
print "\nHeat Map"
sns.heatmap(pcts.reset_index(drop=True), annot=True, cmap='YlGnBu');


# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
from pandas import DataFrame
r2 = []
F = DataFrame(index=data.columns)

for i, col in enumerate(data.columns):
    # print "Considering column '%s' " % col
    y = data[col]
    new_data = data.drop([col], axis=1)

    # TODO: Split the data into training and testing sets using the given feature as the target
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(new_data.values, y, test_size=0.25, random_state=0)

    # TODO: Create a decision tree regressor and fit it to the training set
    from sklearn.tree import DecisionTreeRegressor
    clf = DecisionTreeRegressor(random_state=0)
    clf.fit(X_train, y_train)
    z = pd.Series(clf.feature_importances_, index=new_data.columns, name='Expl. ' + col)
    F = pd.concat([F, z], axis=1)

    # TODO: Report the score of the prediction using the testing set
    r2.append(clf.score(X_test, y_test))
    # print "R2 = ", r2[-1]

r2 = pd.Series(r2, index=data.columns)
print "Table 1.1: Predictability of feature given based on all other features, i.e. 'Fresh' as described by 'Milk', 'Grocery', etc."
display(pd.DataFrame(r2, columns=['R^2']))

corr = data.corr()
for i in range(data.shape[1]):
    for j in range(i,data.shape[1]):
        corr.ix[j, i] = ''
print "\nTable 1.2: Correlation of Features in Original Order"
display(corr)

ix_reord = ['Detergents_Paper', 'Grocery', 'Milk', 'Fresh', 'Frozen', 'Delicatessen']
corr_reord = data[ix_reord].corr()
for i in range(data.shape[1]):
    for j in range(i,data.shape[1]):
        corr_reord.ix[j, i] = ''
print "\nTable 1.3: Correlation of Features in 'Clustered' Order"
display(corr_reord)


corr = data[ix_reord].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, square=True, annot=True, cmap='RdBu')

# display(F)

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# TODO: Scale the data using the natural logarithm
log_data = np.log(data.copy())

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples.copy())

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
pd.scatter_matrix(log_data[ix_reord], alpha = 0.3, figsize = (14,8), diagonal = 'kde');

corr_reord_log = log_data[ix_reord].corr()
for i in range(log_data.shape[1]):
    for j in range(i,log_data.shape[1]):
        corr_reord_log.ix[j, i] = ''
print "\nCorrelation of Features in 'Clustered' Order (log)"
display(corr_reord_log)

# Display the log-transformed sample data
display(log_samples)


out_liers = []

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    print "Feature '%s'" % feature
    x = log_data[feature]

    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(x, 25)
    print "Q1 = %1.2f" % Q1
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(x, 75)
    print "Q3 = %1.2f" % Q3
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5*(Q3 - Q1)
    print "step = %1.2f" % step
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    df_outlier = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    #display(df_outlier) # switching this off for better graphical output
    print "Switching this off for better graphical output\n"

    # OPTIONAL: Select the indices for data points you wish to remove
    for i in df_outlier.index:
        out_liers.append(i)

s = Series(out_liers) # convert outliers into a Series object
s_vc = s.value_counts() # use value_counts method to group by same outlier index
valid = [i for i in range(log_data.shape[0]) if not(i in s_vc[s_vc>1])] # keep all indices that have at most 1 outlier feature

# Remove the outliers, if any were specified
# good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
good_data = log_data.ix[valid,:]

print "Potential outliers = ", len(s_vc)
print "'True' outliers"
print s_vc[s_vc>1]

print "Original data = ", log_data.shape[0]
print "Data without outliers = ", good_data.shape[0]



# TODO: Apply PCA to the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components=data.shape[1])
pca_all = pca.fit(good_data)

# TODO: Apply a PCA transformation to the sample log-data
pca_samples = pca_all.transform(log_samples)

# Generate PCA results plot
# print type(rs)
pca_results = rs.pca_results(good_data, pca_all)
pca_results

print "Cumulative explained variance\n"
print pca_results['Explained Variance'].cumsum()
print "\n"


## Check that data is de-meaned

#good_data_demean = good_data - good_data.mean()
#pca_results_demean = rs.pca_results(good_data_demean, pca)
#pca_results_demean

# The parameter 'whiten' in PCA will control the rescaling of data by their variance (really n_samples * singular values)
# source: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# In this case the variances are close (see below) so I will keep the pca 'whiten' to its default value

    
print "Most important factors (by row) explaining a particular feature (by column)"
display(F)

print "\nStandard deviation of log features"
print log_data.std()

# This gives me an error:
# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(log_samples, 4)))
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# TODO: Fit PCA to the good data using only two dimensions
pca = PCA(n_components=2)
pca_all2 = pca.fit(good_data)

# TODO: Apply a PCA transformation the good data
reduced_data = pca_all2.transform(good_data)

# TODO: Apply a PCA transformation to the sample log-data
pca_samples  = pca_all2.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

from matplotlib import pyplot as pl
pl.scatter(reduced_data.ix[:,0], reduced_data.ix[:,1]);
pl.xlabel('PC1');
pl.ylabel('PC2');
pl.axhline(0, color='k', linestyle='--');
pl.axvline(0, color='k', linestyle='--');


# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# TODO: Apply your clustering algorithm of choice to the reduced data 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

n_clusters = [2, 3, 4, 5, 6]
nC = len(n_clusters)
score = np.zeros((nC, 1))
clr = ['r', 'b', 'y', 'm', 'c', 'k']

# This is VERY slow on my machine ... weird because in the notebook it's not
# Although by CPU is at 15%
# ok, it was the n_job=-1
for i, n_cl in enumerate(n_clusters):
    print "Fitting with # clusters = %i" % n_cl
    clf = KMeans(init='k-means++', n_clusters=n_cl, n_jobs=1)
    clf.fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clf.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clf.cluster_centers_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clf.predict(pca_samples)

    for j in range(n_cl):
        ix = preds==j
        pl.scatter(reduced_data.ix[ix,0], reduced_data.ix[ix,1], color=clr[j])

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score[i] = silhouette_score(reduced_data, clf.labels_)
    print "Score (# clusters = %i) = %1.2f" % (n_cl, score[i])


# Pick best silhoutte score
n_cl = 2

print "Fitting with # clusters = %i" % n_cl
clf = KMeans(init='k-means++', n_clusters=n_cl, n_jobs=1)
clf.fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clf.predict(reduced_data)

# TODO: Find the cluster centers
centers = clf.cluster_centers_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clf.predict(pca_samples)

for j in range(n_cl):
    ix = preds==j
    pl.scatter(reduced_data.ix[ix,0], reduced_data.ix[ix,1], color=clr[j])
pl.plot(centers[0,0], centers[0,1], 'yo', markersize=20)
pl.plot(centers[1,0], centers[1,1], 'go', markersize=20)
pl.xlabel('PC1');
pl.ylabel('PC2');
pl.axhline(0, color='k', linestyle='--');
pl.axvline(0, color='k', linestyle='--');

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score_opt = silhouette_score(reduced_data, clf.labels_)
print "Score (# clusters = %i) = %1.2f" % (n_cl, score_opt)

# Display the results of the clustering from implementation
rs.cluster_results(reduced_data, preds, centers, pca_samples)


# --------------------------------
# GMM
# --------------------------------
from sklearn import mixture
clfGMM = mixture.GMM(n_components=2,covariance_type='full')
aicGMM = np.zeros_like(score)
bicGMM = np.zeros_like(score)
scoreGMM = np.zeros_like(score)
for i, n_cl in enumerate(n_clusters):
    print "Fitting with # clusters = %i" % n_cl
    clfGMM = mixture.GMM(n_components=n_cl, covariance_type='full')
    clfGMM.fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clfGMM.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clfGMM.means_
    covars  = clfGMM.covars_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clfGMM.predict(pca_samples)

    for j in range(n_cl):
        ix = preds==j
        pl.scatter(reduced_data.ix[ix,0], reduced_data.ix[ix,1], color=clr[j])
    pl.show()

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    # Find out how you get the classification, is it via prob_a?
    scoreGMM[i] = silhouette_score(reduced_data, preds)
    print "Score (# clusters = %i) = %1.2f" % (n_cl, scoreGMM[i])
    aicGMM[i] = clfGMM.aic(reduced_data)
    bicGMM[i] = clfGMM.bic(reduced_data)
    print "AIC score (# clusters = %i) = %1.2f" % (n_cl, aicGMM[i])
    print "BIC score (# clusters = %i) = %1.2f" % (n_cl, bicGMM[i])

df_scoreGMM = DataFrame({'aicGMM': aicGMM[:,0], 'bicGMM': bicGMM[:,0]}, index=n_clusters)
display(df_scoreGMM)
df_scoreGMM.plot()

# You want minimum AIC or BIC


# Choose optimal

n_cl_optGMM = 2
clfGMM = mixture.GMM(n_components=n_cl_optGMM, covariance_type='full')
clfGMM.fit(reduced_data)
preds = clfGMM.predict(reduced_data)
centers = clfGMM.means_
covars  = clfGMM.covars_


# Display data

# Scatter
from matplotlib.patches import Ellipse
ax = pl.subplot(111, aspect='equal')
for j in range(n_cl_optGMM):
    ix = preds==j
    ax.scatter(reduced_data.ix[ix,0], reduced_data.ix[ix,1], color=clr[j])
    lam2, v = np.linalg.eig(covars[:,:,j])
    lam = np.sqrt(lam2)
    for k in [1, 2, 4, 6]:
        ell = Ellipse(xy=centers[j,:],
                  width=k*lam2[0], height=k*lam[1],
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  lw=1, color=clr[j])
        ell.set_facecolor('none')
        ax.add_artist(ell)


pl.plot(centers[0,0], centers[0,1], 'yo', markersize=10)
pl.plot(centers[1,0], centers[1,1], 'go', markersize=10)
pl.xlabel('PC1');
pl.ylabel('PC2');
pl.axhline(0, color='k', linestyle='--');
pl.axvline(0, color='k', linestyle='--');


# Adding DBSCAN analysis
from sklearn.cluster import DBSCAN
#clsDB = DBSCAN(eps=0.3, min_samples=10)
clsDB = DBSCAN(eps=0.5, min_samples=10) # manually optimized on eps
cls_fit = clsDB.fit(reduced_data)
scoreDB = silhouette_score(reduced_data, cls_fit.labels_)
print "Score (# clusters = %i) = %1.2f" % (len(set(cls_fit.labels_)), scoreDB)

# Plot
ax = pl.subplot(111, aspect='equal')
labs = cls_fit.labels_ - cls_fit.labels_.min()
labsU = set(labs)
for j in range(len(labsU)):
    ix = labs==j
    ax.scatter(reduced_data.ix[ix,0], reduced_data.ix[ix,1], color=clr[j])

# Ok, kinda. But in this case I really don't like the result



# TODO: Inverse transform the centers
log_centers = pca_all2.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
means = np.round(data.mean())
med = np.round(data.median())
print "\nMean"
display(means)
print "\nMedian"
display(med)
print "\nTotal spending:"
display(np.round(true_centers.sum(axis=1)))

print "\nSpending compared to median"
display(true_centers - med)

print "\nSpending compared to mean"
display(true_centers - means)


# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred


# Display the clustering results based on 'Channel' data
outliers = [154, 66, 75, 128, 65] # from analysis above
rs.channel_results(reduced_data, outliers, pca_samples)
#rs.channel_results(reduced_data, list(out_liers), pca_samples)


# ICA

from sklearn.decomposition import FastICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(good_data)
type(S_) # np array
S_.shape
A_ = ica.mixing_  # Get estimated mixing matrix
A_.shape

ax = pl.subplot(111)
ax.scatter(S_[:,0], S_[:,1])

pd.scatter_matrix(DataFrame(S_), alpha = 0.3, figsize = (5,4), diagonal = 'kde');

# Can you run a clustering on that?
for i, n_cl in enumerate(n_clusters):
    print "Fitting with # clusters = %i" % n_cl
    clf = KMeans(init='k-means++', n_clusters=n_cl, n_jobs=1)
    clf.fit(S_)

    # TODO: Predict the cluster for each data point
    preds = clf.predict(S_)

    # TODO: Find the cluster centers
    centers = clf.cluster_centers_

#    # TODO: Predict the cluster for each transformed sample data point
#    sample_preds = clf.predict(pca_samples)

    for j in range(n_cl):
        ix = preds==j
        pl.scatter(S_[ix,0], S_[ix,1], color=clr[j])

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score[i] = silhouette_score(S_, clf.labels_)
    print "Score (# clusters = %i) = %1.2f" % (n_cl, score[i])


# Pick best silhoutte score
n_cl = 2

print "Fitting with # clusters = %i" % n_cl
clf = KMeans(init='k-means++', n_clusters=n_cl, n_jobs=1)
clf.fit(S_)

# TODO: Predict the cluster for each data point
preds = clf.predict(S_)

# TODO: Find the cluster centers
centers = clf.cluster_centers_

## TODO: Predict the cluster for each transformed sample data point
#sample_preds = clf.predict(pca_samples)

for j in range(n_cl):
    ix = preds==j
    pl.scatter(S_[ix,0], S_[ix,1], color=clr[j])
pl.plot(centers[0,0], centers[0,1], 'yo', markersize=20)
pl.plot(centers[1,0], centers[1,1], 'go', markersize=20)
pl.xlabel('IC1');
pl.ylabel('IC2');
pl.axhline(0, color='k', linestyle='--');
pl.axvline(0, color='k', linestyle='--');


# Compare labels
print "Fitting with # clusters = %i" % n_cl

clf = KMeans(init='k-means++', n_clusters=n_cl, n_jobs=1)
fitPCA = clf.fit(reduced_data)
predsPCA = clf.predict(reduced_data)
centersPCA = clf.cluster_centers_
lblPCA = clf.labels_

fitICA = clf.fit(S_)
predsICA = clf.predict(S_)
centersICA = clf.cluster_centers_
lblICA = clf.labels_


df_cmp = DataFrame(np.c_[lblPCA, lblICA], columns=['PCA', 'ICA'])

ax = pl.subplot(111)
ax.step(range(len(lblPCA)), lblPCA)
ax.step(range(len(lblICA)), lblICA, color='r')

# You can't see anything
ax = pl.subplot(111)
for j in [0, 1]:
    ix = lblPCA==j
    print(len(ix))
    ax.scatter(reduced_data.ix[ix,0], reduced_data.ix[ix,1], color=clr[j], alpha=0.1)
    ix = lblICA==j
    print(len(ix))
    ax.scatter(reduced_data.ix[ix,0], reduced_data.ix[ix,1], marker='+', color=clr[j+2], alpha=0.8, s=80)

ax = pl.subplot(111)
for i in range(len(lblPCA)):
    if lblPCA[i]==lblICA[i]:
        ax.plot(reduced_data.ix[i,0], reduced_data.ix[i,1], 'o', color=clr[lblPCA[i]])
    else:
        ax.scatter(reduced_data.ix[i,0], reduced_data.ix[i,1], s=150, marker='o', color=[0, 0, 0])

# Ok, so IDA classifies differently the points close to the boundary

# Compare the coordinates of PCA with IDA

pca = PCA(n_components=2)
pca_all2 = pca.fit(good_data)

pca_all2.components_[0,:]
ax = pl.subplot(111)
ax.bar(range(6), pca_all2.components_[0,:])
ax = pl.subplot(1,2,1)
ax.bar(range(6), A_[:,0])
ax = pl.subplot(1,2,2)
ax.bar(range(6), A_[:,1])

# Qualitatively very similar to PCA
# In this context ...
