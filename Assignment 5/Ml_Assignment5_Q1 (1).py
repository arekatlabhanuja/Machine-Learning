#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[8]:


spiral_cluster=pd.read_csv('Downloads\spiralwithcluster.csv',usecols=['x','y','SpectralCluster'])

1) a)What percent of the observations have SpectralCluster equals to 1?
# In[9]:


Total_no=spiral_cluster.shape[0]
spectralcluster_rows=spiral_cluster[spiral_cluster['SpectralCluster']==1].shape[0]
percent_of_obs=(spectralcluster_rows/Total_no)*100
print("The percent of observations where spectral cluster equals to 1 are " +str(percent_of_obs))

b) You will search for the neural network that yields the lowest loss value and the lowest misclassification rate.  You will use your answer in (a) as the threshold for classifying an observation into SpectralCluster = 1. Your search will be done over a grid that is formed by cross-combining the following attributes: (1) activation function: identity, logistic, relu, and tanh; (2) number of hidden layers: 1, 2, 3, 4, and 5; and (3) number of neurons: 1 to 10 by 1.  List your optimal neural network for each activation function in a table.  Your table will have four rows, one for each activation function.  Your table will have six columns: (1) activation function, (2) number of layers, (3) number of neurons per layer, (4) number of iterations performed, (5) the loss value, and (6) the misclassification rate.
# In[14]:


def Build_NN_Toy (Layer, nHiddenNeuron,Activation_fun):

    # Build Neural Network
    #min_loss = float("inf")
    #min_misclassification_rate = float("inf")
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*Layer,
                            activation = Activation_fun, verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20200408)
    thisFit = nnObj.fit(spiral_cluster[['x','y']],spiral_cluster[['SpectralCluster']])
    y_pred = nnObj.predict_proba(spiral_cluster[['x','y']])
    Loss = nnObj.loss_
    n_iter = nnObj.n_iter_
    target_class=spiral_cluster['SpectralCluster']
    #~no_target=target_class.shape[0]

    # Threshold filtering
    predict_data=np.empty_like(target_class)
    threshold=percent_of_obs
    for i in range(len(target_class)):
        if (y_pred[i][0]>=0.5):
            predict_data[i] = 0
        else:
            predict_data[i] = 1
    
    # Accuracy and misclassification rate
    accuracy=metrics.accuracy_score(target_class,predict_data)
    misclassification_rate=1-accuracy
    
    
    
    return(Loss,misclassification_rate,Activation_fun,n_iter)

    


# In[24]:


warnings.filterwarnings("ignore")
act = ['identity','logistic','relu','tanh']
result = pd.DataFrame(
    columns=['Index','nLayer', 'nHiddenNeuron','No_iterations', 'Loss','misclassification_rate', 'Activation_fun'])
#result2 = pd.DataFrame(
#    columns=['Index','nLayer', 'nHiddenNeuron', 'Activation_fun', 'Loss','misclassification_rate','No_iterations'])

index=0
for k in act:
    min_loss = float('inf')
    min_misclassification_rate = float('inf')
    for i in np.arange(1,6):
        for j in np.arange(1,11,1):
            Loss,misclassification_rate, Activation_fun,n_iter = Build_NN_Toy (Layer = i, nHiddenNeuron = j,Activation_fun = k)
            # Minimum loss and miclassification rate
            if (Loss <= min_loss and misclassification_rate <=min_misclassification_rate):
                min_loss = Loss
                min_misclassification_rate = misclassification_rate
                Layer = i
                nHiddenNeuron = j
                ter = n_iter
                func = k
            #result2 = result2.append(pd.DataFrame([[index,i, j,Loss,misclassification_rate,k, n_iter]], 
             #                  columns = [ 'Index','nLayer', 'nHiddenNeuron', 'Loss','misclassification_rate', 'Activation_fun','No_iterations']))

    result = result.append(pd.DataFrame([[index,Layer, nHiddenNeuron, ter,min_loss, min_misclassification_rate,func]], 
                               columns = [ 'Index','nLayer', 'nHiddenNeuron','No_iterations', 'Loss','misclassification_rate', 'Activation_fun']))
    index+=1
pd.set_option('display.max_rows', result.shape[0]+1)
pd.set_option('display.max_columns', result.shape[1]+1)
result = result.set_index('Index')
print(result)
print(result[result.Loss == result.Loss.min()])


# In[16]:


result

d)(5 points) Which activation function, number of layers, and number of neurons per layer give the lowest loss and the lowest misclassification rate?  What are the loss and the misclassification rate?  How many iterations are performed?
# In[20]:


df=pd.DataFrame(result)
Min_value=df.loc[df['Loss'].idxmin()]
print(Min_value)

c)What is the activation function for the output layer?, e)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue) from the optimal MLP in (d).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.
# In[21]:


nnObj = nn.MLPClassifier(hidden_layer_sizes = (10,)*4,
                            activation ='relu', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20200408)
thisFit = nnObj.fit(spiral_cluster[['x','y']],spiral_cluster['SpectralCluster'])
y_pred = nnObj.predict_proba(spiral_cluster[['x','y']])
spiral_cluster['y_pred'] = y_pred[:,1]
target_class=spiral_cluster['SpectralCluster']
    #~no_target=target_class.shape[0]

    # Threshold filtering
predict_data=np.empty_like(target_class)
for i in range(len(target_class)):
    if (y_pred[i][0]>=0.5):
        predict_data[i] = 0
    else:
        predict_data[i] = 1
spiral_cluster['pred'] = predict_data   
    # Accuracy and misclassification rate
accuracy=metrics.accuracy_score(target_class,predict_data)
misclassification_rate=1-accuracy
print('The activation function for the output layer is ',nnObj.out_activation_)


# In[22]:


carray = ['red', 'blue']
for i in range(2):
    subData = spiral_cluster[spiral_cluster['pred'] == i]
    plt.scatter(x = subData['x'],y=subData['y'],c = carray[i], label = i, s = 25)
plt.title('x coordinate versus y coordinate')
plt.grid(True)
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend(title='predicted Cluster')
plt.show()

f)	(5 points) What is the count, the mean and the standard deviation of the predicted probability Prob(SpectralCluster = 1) from the optimal MLP in (d) by value of the SpectralCluster?  Please give your answers up to the 10 decimal places.
# In[23]:


pd.set_option('display.max_rows', 9)
print(spiral_cluster[spiral_cluster['pred']==1]['y_pred'].describe())


# In[ ]:




