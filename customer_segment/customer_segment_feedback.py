# -*- coding: utf-8 -*-
"""
Created on Sat May 21 14:43:33 2016

@author: rghiglia
"""

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from sklearn import mixture
import numpy as np

colors = [(1,'blue'),(2,'red'),(3,'orange'),(4,'yellow'),(5,'purple')]

clusters = []
for n,color in colors:
    cluster = multivariate_normal(
        mean = np.array([5*n,0]),
        cov = np.array([[0.4,0],[0,3.13]])).rvs(100)

    if n==1:
        X = cluster
    else:
        X = np.vstack((X,cluster))

    clusters.append(cluster)

g = mixture.GMM(n_components=5,covariance_type='full')
g.fit(X)

centroids = g.means_
covars  = g.covars_

#Plot an ellipses
for i in range(len(centroids)):

    mean = centroids[i]
    cov = covars[i]

    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ax = plt.subplot(111, aspect='equal')

    ell = Ellipse(xy=mean,
              width=lambda_[0], height=lambda_[1],
              angle=np.rad2deg(np.arccos(v[0, 0])))

    ell.set_facecolor('none')
    ax.add_artist(ell)

#Plot data
plt.title('GMM')
for i,color in colors:
    cluster = clusters[i-1]
    plt.scatter([c[0] for c in cluster],[c[1] for c in cluster],color=color)


#Plot centroids
plt.scatter([c[0] for c in centroids],
    [c[1] for c in centroids],color='black')

plt.show()



# Non-convex clusters

c1 = multivariate_normal.rvs(mean=[0,2],cov=0.1,size=100)
c1 = np.vstack((c1,multivariate_normal.rvs(mean=[0,1],cov=0.1,size=100)))
c1 = np.vstack((c1,multivariate_normal.rvs(mean=[0,0],cov=0.1,size=100)))
c1 = np.vstack((c1,multivariate_normal.rvs(mean=[1,0],cov=0.1,size=100)))
c1 = np.vstack((c1,multivariate_normal.rvs(mean=[2,0],cov=0.1,size=100)))
c1 = np.vstack((c1,multivariate_normal.rvs(mean=[3,0],cov=0.1,size=100)))

c2 = multivariate_normal.rvs(mean=[1.5,2],cov=0.1,size=100)
c2 = np.vstack((c2,multivariate_normal.rvs(mean=[2.5,2],cov=0.1,size=100)))
c2 = np.vstack((c2,multivariate_normal.rvs(mean=[3.5,2],cov=0.1,size=100)))

X = np.vstack((c1,c2))

g = mixture.GMM(n_components=2,covariance_type='full')
g.fit(X)

centroids = g.means_
covars  = g.covars_

#Plot an ellipses
for i in range(len(centroids)):

    mean = centroids[i]
    cov = covars[i]

    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ax = plt.subplot(111, aspect='equal')

    ell = Ellipse(xy=mean,
              width=lambda_[0], height=lambda_[1],
              angle=np.rad2deg(np.arccos(v[0, 0])),color='black')

    ell.set_facecolor('none')
    ax.add_artist(ell)

#Plot data
plt.title('GMM on non-convex clusters')
plt.scatter([c[0] for c in c1],[c[1] for c in c1],color='orange')
plt.scatter([c[0] for c in c2],[c[1] for c in c2],color='green')


#Plot centroids
plt.scatter([c[0] for c in centroids],
    [c[1] for c in centroids],color='black')
