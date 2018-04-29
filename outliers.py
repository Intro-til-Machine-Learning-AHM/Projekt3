import numpy as np
import pandas as pd
import toolbox_02450 as tb
from math import inf
import matplotlib.pyplot as plot
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = pd.read_csv('data.csv')
X = data.drop("class",axis=1)
X = scaler.fit_transform(X.as_matrix())

best_logP = -inf
best_bandwidth = None
best_density = None

for bandwidth in np.linspace(0,5, 1000):
    density, log_density = tb.gausKernelDensity(X, bandwidth)
    logP = log_density.sum()
    if logP > best_logP:
        best_logP = logP
        best_bandwidth = bandwidth
        best_density = density

kde_density = best_density[:,0]

# Number of neighbors
K = 200

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Compute the density
knn_density = 1./(D.sum(axis=1)/K)

# Compute the average relative density
knn_densityX = 1./(D[:,1:].sum(axis=1)/K)
knn_avg_rel_density = knn_density/(knn_densityX[i[:,1:]].sum(axis=1)/K)

kde_sort = np.argsort(kde_density)
kde_density.sort()

knn_sort = np.argsort(knn_density)
knn_density.sort()

knn_avg_rel_sort = np.argsort(knn_avg_rel_density)
knn_avg_rel_density.sort()


plot.figure()
plot.title('KDE densities')
plot.bar(range(X.shape[0]), kde_density, width = 1)
plot.savefig("pics/kde_densities.png")

plot.figure()
plot.title('KNN densities')
plot.bar(range(X.shape[0]), knn_density, width = 1)
plot.savefig("pics/knn_densities.png")

plot.figure()
plot.title('KNN average relative densities')
plot.bar(range(X.shape[0]), knn_avg_rel_density, width = 1)
plot.savefig("pics/knn_avg_rel_densities.png")

