#!/usr/bin/env python
# coding: utf-8

# In[156]:


import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as s
from collections import Counter
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
import numpy.linalg as linalg
import math
import sklearn.neighbors as neighbors
from sklearn.neighbors import NearestNeighbors as kNN

1)a)	(5 points) Create a data frame that contains the number of unique items in each customer’s market basket. Draw a histogram of the number of unique items.  What are the 25th, 50th, and the 75th percentiles of the histogram?
# In[270]:


groceries_data=pd.read_csv("Groceries.csv")
groceries_data.head()
unique_items = groceries_data.groupby(['Customer'])['Item'].nunique()
s.distplot(unique_items)
plt.show()
firstperc=float(desc.loc['25%'])
med=float(desc.loc['50%'])
secondperc=float(desc.loc['75%'])
print('The 25th percentile of the histogram  :',firstperc)
print('The median of this histogram is: ',med)
print('The 75 percentile of this histogram is :',secondperc)

b) We are only interested in the k-itemsets that can be found in the market baskets of at least seventy five (75) customers.  How many itemsets can we find?what is the largest k value among our itemsets?
# In[269]:


Itemlist = groceries_data.groupby(['Customer'])['Item'].apply(list).values.tolist()
t = TransactionEncoder()
tr_ary = t.fit(Itemlist).transform(Itemlist)

df = pd.DataFrame(tr_ary, columns=t.columns_)
tot_Transactions=np.count_nonzero(unique_items)
frequent_itemsets = apriori(df, min_support =(75/tot_Transactions), max_len=med,use_colnames = True)


print ("Frequent itemsets are: \n",frequent_itemsets.head())
Total_no_of_freq=frequent_itemsets['itemsets'].value_counts().sum()
print("Total number of frequeny itemsets",Total_no_of_freq)
k_value = len(frequent_itemsets['itemsets'][len(frequent_itemsets)-1])
print("The highest value of k in the itemset:",k_value)

C)Find out the association rules whose Confidence metrics are greater than or equal to 1%.  How many association rules can we find?  Please be reminded that a rule must have a non-empty antecedent and a non-empty consequent
# In[20]:


assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("No of associaton rules are:",len(assoc_rules))

d)Plot the Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for the rules you have found in (c).  Please use the Lift metrics to indicate the size of the marker.
# In[26]:


print("Scatter plot for support and confidence metrics")
plt.figure(figsize=(8,8))
s.scatterplot(data=assoc_rules,x="confidence",y="support",size="lift")
plt.show()

e)	(5 points) List the rules whose Confidence metrics are greater than or equal to 60%.  Please include their Support and Lift metrics.
# In[29]:


rules=association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.60)
print("Rules whose confidence metrics are greater than or equal to 60%")
display(rules)


# In[ ]:


get_ipython().set_next_input('2)a)What are the frequencies of the categorical feature Type');get_ipython().run_line_magic('pinfo', 'Type')


# In[172]:


cars_data=pd.read_csv("cars.csv")
cars_loc=cars_data.loc[:,['Type','Origin','DriveTrain','Cylinders']]
cars_data.head()


# In[274]:


x=cars_data.Type.value_counts()
y=list(zip(cars_data['Type'].value_counts().index,x))
df = pd.DataFrame(y,columns=['Type','Counts'])
print(df)


# In[ ]:


get_ipython().set_next_input('b)What are the frequencies of the categorical feature DriveTrain');get_ipython().run_line_magic('pinfo', 'DriveTrain')


# In[275]:


x1=cars_data.DriveTrain.value_counts()
y1=list(zip(cars_data['DriveTrain'].value_counts().index,x1))
df1=pd.DataFrame(y1,columns=['DriveTrain','Counts'])
print(df1)

c)What is the distance metric between ‘Asia’ and ‘Europe’ for Origin?
# In[276]:


x=cars_data.Origin.value_counts()['Asia']
y=cars_data.Origin.value_counts()['Europe']
dist_metric_origin=(1/x)+(1/y)
print(round(dist_metric_origin,7))

d)What is the distance metric between Cylinders = 5 and Cylinders = Missing?
# In[179]:


Z=cars_data.Cylinders.value_counts()[5.0]
Q=cars_data.Cylinders.value_counts(dropna=False)[np.NaN]
dist_metric_cylinders=(1/Z)+(1/Q)
print(round(dist_metric_cylinders,7))

e)Apply the K-modes method with three clusters.  How many observations in each of these three clusters?  What are the centroids of these three clusters?
# In[277]:


km = KModes(n_clusters=3, init='Cao', max_iter=15)
clusters = km.fit_predict(cars_loc)
unique, counts = np.unique(km.labels_, return_counts=True)
num_ele=dict(zip(unique, counts))
print(num_ele)
# Print the cluster centroids
print(km.cluster_centroids_)

f)Display the frequency distribution table of the Origin feature in each cluster.
# In[264]:


freq_dist = pd.DataFrame(list(zip(clusters,cars_data['Origin'])),columns=['Cluster','Origin'])
f = freq_dist.groupby(['Cluster','Origin']).size()
print(f)

3)a)Plot y on the vertical axis versus x on the horizontal axis.  How many clusters are there based on your visual inspection?
# In[112]:


fourcircle_data=pd.read_csv("FourCircle.csv")
fourcircle_data.head()


# In[120]:


plt.figure(figsize=(5,5))
s.scatterplot(data=fourcircle_data,x='x',y='y')
plt.grid()
plt.show()
print("By visual inspection we can observe 4 clusters")

b) Apply the K-mean algorithm directly using your number of clusters that you think in (a). Regenerate the scatterplot using the K-mean cluster identifiers to control the color scheme
# In[196]:


import sklearn.cluster as cluster 

val=fourcircle_data[['x','y']]
k_means_algo=cluster.KMeans(n_clusters=4,random_state=60616)
fit_val=k_means_algo.fit(val)
plt.figure(figsize=[5,5])
s.scatterplot(x='x',y='y',data=fourcircle_data,hue=fit_val.labels_)
plt.show()

c)Apply the nearest neighbor algorithm using the Euclidean distance.  We will consider the number of neighbors from 1 to 15.  What is the smallest number of neighbors that we should use to discover the clusters correctly?

d)Using your choice of the number of neighbors in (c), calculate the Adjacency matrix, the Degree matrix, and finally the Laplacian matrix. How many eigenvalues do you determine are practically zero?  Please display values of the “zero” eigenvalues in scientific notation.
# In[249]:


import scipy 
# Fourteen nearest neighbors8
kNNSpec = neighbors.NearestNeighbors(n_neighbors = 6 ,
   algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(val)
d3, i3 = nbrs.kneighbors(val)
nObs = fourcircle_data.shape[0]

# Retrieve the distances among the observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(val)

# Create the Adjacency matrix
Adjacency = np.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )

# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())
print("Adjacency Matrix: \n",Adjacency)

# Create the Degree matrix
Degree = np.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
print("Degree Matrix: \n",Degree)

# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency
print("Laplacian Matrix: \n",Lmatrix)
# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = linalg.eigh(Lmatrix)
for j in range(10):
    print('Eigenvalue: ', evals[j])
Z = evecs[:,[0,1]]
plt.scatter(1e10*Z[:,0], Z[:,1])
plt.xlabel('First Eigenvector')
plt.ylabel('Second Eigenvector')
plt.grid(True)
plt.show()

sequence = np.arange(1,6,1) 
plt.plot(sequence,evals[0:5,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.xticks(sequence)
plt.grid(True)
plt.show()

e)	(10 points) Apply the K-mean algorithm on the eigenvectors that correspond to your “practically” zero eigenvalues.  The number of clusters is the number of your “practically” zero eigenvalues. Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme.
# In[181]:


kmeans_spectra = cluster.KMeans(n_clusters = 3, random_state = 0).fit(Z)
fourcircle_data['SpectralCluster'] = kmeans_spectral.labels_
plt.scatter(fourcircle_data['x'], fourcircle_data['y'], c = fourcircle_data['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


0


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




