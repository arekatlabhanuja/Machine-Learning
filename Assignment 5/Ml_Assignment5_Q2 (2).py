#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.svm as svm
import warnings

import seaborn as sns


# In[3]:


warnings.filterwarnings("ignore")
spiral_cluster=pd.read_csv('Downloads/SpiralWithCluster.csv',delimiter=',',usecols=['x','y','SpectralCluster'])
x_train=spiral_cluster[['x','y']]
y_train=spiral_cluster['SpectralCluster']

a)	(5 points) What is the equation of the separating hyperplane?  Please state the coefficients up to seven decimal places.
# In[5]:


#Ref from slides
svm_model = svm.SVC(kernel='linear', decision_function_shape='ovr', random_state=20200408, max_iter=-1)
this_fit = svm_model.fit(x_train, y_train)
y_pred=svm_model.predict(x_train)
spiral_cluster['pred'] = y_pred
intercept=np.round(this_fit.intercept_,7)
coefficient=np.round(this_fit.coef_, 7)
print('Intercept =' ,intercept)
print('Coefficients =' ,coefficient)
print('The equation separating hyperplane is {} + ({} x1) + ({} x2) = 0 '.format(intercept[0],coefficient[0][0],coefficient[0][1]))

b)(5 points) What is the misclassification rate?
# In[5]:


accuracy=metrics.accuracy_score(y_pred,y_train)
miscla=1-accuracy
print('The misclassification rate is',miscla)

c)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue).  Besides, plot the hyperplane as a dotted line to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.
# In[6]:



w = this_fit.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5,5)
yy = a * xx - ((this_fit.intercept_[0]) / w[1])
#sns.scatterplot(x='x',y='y',hue='pred',data=spiral_cluster)
carray = ['red', 'blue']
for i in range(2):
    subData = spiral_cluster[spiral_cluster['pred'] == i]
    plt.scatter(x = subData['x'],y=subData['y'],c = carray[i], label = i, s = 25)
plt.title('SVM Classifier Plot')
plt.grid(True)
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend(title='predicted Cluster')
plt.plot(xx,yy,'k:')
plt.show()

d)	(10 points) Please express the data as polar coordinates.  Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster variable (0 = Red and 1 = Blue).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.
# In[7]:


# Convert to the polar coordinates
def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)

spiral_cluster['radius'] = np.sqrt(spiral_cluster['x']**2 + spiral_cluster['y']**2)
spiral_cluster['theta'] = np.arctan2(spiral_cluster['y'], spiral_cluster['x'])


spiral_cluster['theta'] = spiral_cluster['theta'].apply(customArcTan)

# Scatterplot that uses prior information of the grouping variable
carray = ['red', 'blue']
plt.figure(figsize=(7,7))
for i in range(2):
    subData = spiral_cluster[spiral_cluster['SpectralCluster'] == (i)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = (i), s = 25)
plt.grid(True)
plt.title('Prior Group Information')
plt.xlabel('Radius')
plt.ylabel('Angle in Radians')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

        e)	(10 points) You should expect to see three distinct strips of points and a lone point.  Since the SpectralCluster variable has two values, you will create another variable, named Group, and use it as the new target variable. The Group variable will have four values. Value 0 for the lone point on the upper left corner of the chart in (d), values 1, 2,and 3 for the next three strips of points.
Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[8]:


group = np.zeros(spiral_cluster.shape[0])

# create four group by using the location of the coordinates
for index, row in spiral_cluster.iterrows():
    if row['radius'] < 1.5 and row['theta'] > 6:
        group[index] = 0
    elif row['radius'] < 2.5 and row['theta'] > 3:
        group[index] = 1
    elif 2.5 < row['radius'] < 3 and row['theta'] > 5.5:
        group[index] = 1
    elif row['radius'] < 2.5 and row['theta'] < 3:
        group[index] = 2
    elif 3 < row['radius'] < 4 and 3.5 < row['theta'] < 6.5:
        group[index] = 2
    elif 2.5 < row['radius'] < 3 and 2 < row['theta'] < 4:
        group[index] = 2
    elif 2.5 < row['radius'] < 3.5 and row['theta'] < 2.25:
        group[index] = 3
    elif 3.55 < row['radius'] and row['theta'] < 3.25:
        group[index] = 3

spiral_cluster['group'] = group
# plot coordinates divided into four group
color_array = ['red', 'blue', 'green', 'black']
for i in range(4):
    x_y = spiral_cluster[spiral_cluster['group'] == i]
    plt.scatter(x=x_y['radius'], y=x_y['theta'], c=color_array[i], label=i)
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.title('SVM for Radius Coordinate versus Theta Coordinate')
plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()

f)	(10 points) Since the graph in (e) has four clearly separable and neighboring segments, we will apply the Support Vector Machine algorithm in a different way.  Instead of applying SVM once on a multi-class target variable, you will SVM three times, each on a binary target variable.
SVM 0: Group 0 versus Group 1
SVM 1: Group 1 versus Group 2
SVM 2: Group 2 versus Group 3
Please give the equations of the three hyperplanes
g)(5 points) Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black). Please add the hyperplanes to the graph. To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.
# In[9]:


# build SVM 0: Group 0 versus Group 1
yy_h = []
for i in range (3):
    subi = spiral_cluster[spiral_cluster['group']==i]
    subj = spiral_cluster[spiral_cluster['group']==i+1]
    subij = subi.append(subj)
    x = subij[['radius','theta']]
    y = subij['group']
    svm_model = svm.SVC(kernel='linear',decision_function_shape='ovr',random_state=20200408,max_iter=-1)
    fit_ = svm_model.fit(x,y)
    w = fit_.coef_[0]
    a = -w[0]/w[1]
    xx = np.linspace(1,4)
    yy = a * xx - ((fit_.intercept_[0]) / w[1])
    yy_h.append(yy)
    plt.plot(xx,yy,'k:')
    plt.grid(True)
    inter = fit_.intercept_
    coef = fit_.coef_
    print('The equation separating hyperplane for SVM is  ({}) + ({}*x1) + ({}*x2) = 0'.format(np.round(inter[0],7),np.round(coef[0][0],7),np.round(coef[0][1],7)))

color_array = ['red', 'blue', 'green', 'black']
for i in range(4):
    x_y = spiral_cluster[spiral_cluster['group'] == i]
    plt.scatter(x=x_y['radius'], y=x_y['theta'], c=color_array[i], label=i)
plt.xlabel('Radius Coordinates')
plt.ylabel('Theta Coordinates')
plt.title('SVM for Radius Coordinate versus Theta Coordinate')
plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()

h)(10 points) Convert the observations along with the hyperplanes from the polar coordinates back to the Cartesian coordinates. Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster (0 = Red and 1 = Blue). Besides, plot the hyper-curves as dotted lines to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.
Based on your graph, which hypercurve do you think is not needed?	
# In[10]:


xx1=np.linspace(1,4)
xx2=np.linspace(1,4)
xx3=np.linspace(1,4)
h1_xx1 = xx1 * np.cos(yy_h[0])
h1_yy1 = xx1 * np.sin(yy_h[0])
h2_xx2 = xx2 * np.cos(yy_h[1])
h2_yy2 = xx2 * np.sin(yy_h[1])
h3_xx3 = xx3 * np.cos(yy_h[2])
h3_yy3 = xx3 * np.sin(yy_h[2])
# plot the line, the coordinates, and the nearest vectors to the plane

plt.plot(h1_xx1, h1_yy1, color='green', linestyle=':')
plt.plot(h2_xx2, h2_yy2, color='black', linestyle=':')
plt.plot(h3_xx3, h3_yy3, color='black', linestyle=':')
color_array = ['red', 'blue']
for i in range(2):
    x_y = spiral_cluster[spiral_cluster['SpectralCluster'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.xlabel('x axis of coordinates')
plt.ylabel('y axis of coordinates')
plt.title('Support Vector Machines on Two Segments')
plt.legend(title='Spectral_Cluster', loc='best')
plt.grid(True)
plt.show()
print('I think green hypercurve is not needed, based on my graph')


# In[ ]:





# In[ ]:





# In[ ]:




