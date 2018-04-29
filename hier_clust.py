# exercise 10.2.1
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from scipy import stats
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from toolbox_02450 import clusterval
import pandas as pd

# Load Matlab data file and extract variables of interest
#mat_data = loadmat('..\\..\\..\\02450Toolbox_Python\\Data\\synth1.mat')
data = pd.read_csv('data.csv')
X = data.drop("class",axis=1)
y = data['class']
X = stats.zscore(X.as_matrix())
y = y.as_matrix()
attributeNames = list(X)
classNames = ['Class 1','Class 2']
N, M = X.shape
C = len(classNames)


# Perform hierarchical/agglomerative clustering on data matrix
Method = 'ward'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 2
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
#figure(1)
#clusterplot(X, cls.reshape(cls.shape[0],1), y=y)
figure(figsize=(14,9))
idx = [4,1] # feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, y=y)
#ylabel("glucose")
#xlabel("insulin")
show()
# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()

Rand, Jaccard, NMI = clusterval(y,cls)
print(Rand,Jaccard,NMI)

print('Ran Exercise 10.2.1')
