# exercise 11.1.1
from matplotlib.pyplot import figure, show, ylabel, xlabel
import numpy as np
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GaussianMixture
import pandas as pd

# Load Matlab data file and extract variables of interest
#mat_data = loadmat('..\\..\\..\\02450Toolbox_Python\\Data\\synth1.mat')
data = pd.read_csv('..\\Projekt3\\data.csv')
X = data.drop("class",axis=1)
y = data['class']
X = X.as_matrix()
y = y.as_matrix()
attributeNames = list(X)
classNames = ['Class 1','Class 2']
N, M = X.shape
C = len(classNames)

# Number of clusters
K = 2
cov_type = 'full'
# type of covariance, you can try out 'diag' as well
reps = 1
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X)
cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])

    count = 0
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
#figure(figsize=(14,9))
#clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
#show()
print(cds)

## In case the number of features != 2, then a subset of features most be plotted instead.
figure(figsize=(14,9))
idx = [4,1] # feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
ylabel("glucose")
xlabel("insulin")
show()

print('Ran Exercise 11.1.1')
