#!/usr/bin/env python
# coding: utf-8
1)
a)According to Izenman (1991) method, what is the recommended bin-width for the histogram of x?
# In[285]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as s
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier as kNN


# In[117]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[118]:


df=pd.read_csv("Downloads/NormalSample.csv")


# In[119]:


df.head()


# In[345]:


df.info()
x=df.iloc[:,2]


# In[343]:


Q1=np.percentile(df.x,25)
Q3=np.percentile(df.x,75)
IQR=Q3-Q1
bin_width=2*(IQR)*(len(df))**(-1/3)
print("Q1",Q1)
print("Q3",Q3)
print("IQR=",IQR)
print("bin_width",bin_width)

b) What are the minimum and the maximum values of the field x?
# In[122]:


Minimum_value=df.x.min()
Maximum_value=df.x.max()
print("Minimum value:",Minimum_value);
print("Maximum value:",Maximum_value);

c) Let a be the largest integer less than the minimum value of the field x, and b be the smallest integer greater than the maximum value of the field x.  What are the values of a and b?
# In[123]:


b=np.floor(Minimum_value)
p=np.ceil(Maximum_value)
print("Value of a=",b)
print("Value of b=",p)

d) Use h = 0.25, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools.
# In[124]:


def density_estimation(h_,sample):
    prob_ = []

    mid_ = [b+(h_/2)]
    while True:    
        u=[(sample[i]-mid_[-1])/h_ for i in range(len(sample))]
        w = [1 if -1/2 < j <= 1/2 else 0  for j in u]
        prob_.append(sum(w)/(len(sample)*h_))
        if mid_[-1] >= p:
            break
        else:
            mid_.append(mid_[-1]+h_)

    return prob_[:-1],mid_[:-1]

density_values,m_values = density_estimation(0.25,df['x'])
print('p_values,m_values')
print(list(zip(density_values,m_values)))
plt.figure()
plt.hist(df['x'],bins=int((p-b)/0.25))
plt.show()

e)Use h = 0.5, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools.
# In[125]:


value_of_density,midpoint_value = density_estimation(0.5,df['x'])
print(list(zip(value_of_density,midpoint_value)))

plt.figure()
plt.hist(df['x'],bins=int((p-b)/0.5))
plt.show()

f) Use h = 1, minimum = a and maximum = b. List the coordinates of the density estimator. Paste the histogram drawn using Python or your favorite graphing tools.
# In[126]:


value_of_density,midpoint_value = density_estimation(1,df['x'])
print(list(zip(value_of_density,midpoint_value)))

plt.figure()
plt.hist(df['x'],bins=int((p-b)))
plt.show()

g)Use h = 2, minimum = a and maximum = b. List the coordinates of the density estimator. Paste the histogram drawn using Python or your favorite graphing tools.
# In[127]:


value_of_density,midpoint_value = density_estimation(2,df['x'])
print(value_of_density,midpoint_value)
plt.figure()
plt.hist(df['x'],bins=int((p-b)/2))
plt.show()

2)Use in the NormalSample.csv to generate box-plots for answering the following questions.
a) What is the five-number summary of x?  What are the values of the 1.5 IQR whiskers?

# In[128]:


print(df.describe())


# In[129]:


print(df.x.describe())


# In[130]:


def five_numbsummary(d):
    d=np.array(d)
    minimum_value=d.min()
    maximum_value=d.max()
    quart1,quart2,quart3=np.percentile(d,25),np.percentile(d,50),np.percentile(d,75)
    IQR=quart3-quart1
    L_whisk, R_whisk=quart1-(IQR*1.5),quart3+(IQR*1.5)
    print("Summary of the data:",minimum_value,quart1,quart2,quart3,maximum_value)
    print("IQR Whiskers",L_whisk,R_whisk)
    return L_whisk,quart1,quart2,quart3,R_whisk
    


# In[131]:


print(five_numbsummary)


# In[132]:


xL_whisk,xquart1,xquart2,xquart3,xR_whisk=five_numbsummary(df['x'])

b)What is the five-number summary of x for each category of the group? What are the values of the 1.5 IQR whiskers for each category of the group?
# In[133]:


print(df.group.describe())


# In[134]:


df.head()


# In[135]:


print(type(df))


# In[136]:


g=df["group"]
g1_data=[]
g2_data=[]
k=0
while (k<len(g)):
    if g[k]==0:
        g1_data.append(df.iloc[k,2])
    else:
        g2_data.append(df.iloc[k,2])
    k+=1


# In[137]:


pd.Series(g1_data).describe()


# In[138]:


pd.Series(g2_data).describe()


# In[139]:


print("Whiskers for group 0")
print(list(zip(['Lower Whisker','Upper whisker'],five_numbsummary(g1_data))))


# In[140]:


print("Whiskers for group 1")
print(list(zip(['Lower Whisker','Upper whisker'],five_numbsummary(g2_data))))

c)Draw a boxplot of x (without the group) using the Python boxplot function.  Can you tell if the Python’s boxplot has displayed the 1.5 IQR whiskers correctly?
# In[141]:


sns.boxplot(df['x'])

d)Draw a graph where it contains the boxplot of x, the boxplot of x for each category of Group (i.e., three boxplots within the same graph frame).  Use the 1.5 IQR whiskers, identify the outliers of x, if any, for the entire data and for each category of the group.
# In[355]:


data = pd.DataFrame(list(zip(x,g1_data,g2_data)),columns=['x','Group0','Group1'])
plt.figure()
sns.boxplot(data=data)
plt.show()


# In[346]:


def get_outlier(d):
    j=0
    no_of_outliers = []
    while j<len(d):
        if d[j]<Q1 - (1.5*IQR) or d[j]>Q3 + (1.5*IQR):
            no_of_outliers.append(d[j])
        j+=1
    return no_of_outliers
#outliers
print("Outliers present in data ",get_outlier(x))
print("Outliers present in Group 1 data: ",get_outlier(g1_data))
print("Outliers present in Group 2 data: ",get_outlier(g2_data))

3)
a) What percent of investigations are found to be fraudulent?  Please give your answer up to 4 decimal places.
# In[165]:


fraudulent_data=pd.read_csv("Downloads/Fraud.csv")


# In[166]:


fraudulent_data.head()


# In[167]:


count=0
for f in fraudulent_data['FRAUD']:
    if f==1:
        count+=1
fraud_percentage=(count/fraudulent_data['FRAUD'].count())*100
print("Percentage of investigations which are fraud are:",fraud_percentage)
roundup_value=round(fraud_percentage,4)
print("Round off value of fraud",roundup_value)

b)Use the BOXPLOT function to produce horizontal box-plots.  For each interval variable, one box-plot for the fraudulent observations, and another box-plot for the non-fraudulent observations.  These two box-plots must appear in the same graph for each interval variable.
# In[168]:


fraudulent_data.head()


# In[169]:


cols = fraudulent_data.drop(['FRAUD'],axis=1).columns
for j in list(cols):
    plt.figure()
    sns.boxplot(x=j,y='FRAUD',data=fraudulent_data,orient='h')
    plt.show()

c)Orthonormalize interval variables and use the resulting variables for the nearest neighbor analysis. Use only the dimensions whose corresponding eigenvalues are greater than one.
i.How many dimensions are used?
ii.Please provide the transformation matrix?
# In[356]:


drop_data = fraudulent_data.drop(['FRAUD','CASE_ID'],axis=1)
matrix = np.matrix(drop_data.values)
print("No. of Dimensions which are used{} ".format(len(drop_data.columns)))
print("Input Matix used \n",matrix)
print("No. of Dimensions:",matrix.ndim)
print("No. of Rows: ", np.size(matrix,0))
print("No. of Columns: ", np.size(matrix,1))
trans = matrix.transpose()*matrix
evalues,evectors = LA.eigh(trans)
print("Eigen Values: ",evalues)
print("\nEigen Vectors: ",evectors)
transformation_matrix = evectors * LA.inv(np.sqrt(np.diagflat(evalues)))
print("Transformation Matrix: \n",transformation_matrix)
transformed_matrix = matrix * transformation_matrix
print("The Transformed Matrix: \n",transformed_matrix)
ident_matrix = transformed_matrix.transpose().dot(transformed_matrix)
print("An Identity Matrix: \n",ident_matrix)

from scipy import linalg as LI
ortho_matrix = LI.orth(matrix)
print("The Orthonormalize of matrix: ",ortho_matrix)
c = ortho_matrix.transpose().dot(ortho_matrix)
print("Identity Matrix: \n",c)
print("The Transformed Matrix: \n",ortho_matrix)
print("Variables are orthonormal: \n",c )

d)Use the NearestNeighbors module to execute the Nearest Neighbors algorithm using exactly five neighbors and the resulting variables you have chosen in c).  The KNeighborsClassifier module has a score function.
# In[328]:


kNNS = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
k=np.array(fraudulent_data.loc[:,'FRAUD'])
fitting_matrix=kNNS.fit(matrix,k)
predicted_matrix = fitting_matrix.predict(matrix)
print("Score of the kneighbors: ",fitting_matrix.score(matrix,k))
n = kNNS.fit(transformed_matrix,k)

e)For the observation which has these input variable values: TOTAL_SPEND = 7500, DOCTOR_VISITS = 15, NUM_CLAIMS = 3, MEMBER_DURATION = 127, OPTOM_PRESC = 2, and NUM_MEMBERS = 2, find its five neighbors.  Please list their input variable values and the target values
# In[336]:


f= np.array([7500,15,3,127,2,2])
transformation_focal = f*transformation_matrix
neighbours_data = n.kneighbors(transformation_focal,return_distance=False)
result = fitting_matrix.predict(transformation_focal)
print(drop_data.columns)
print("Neighbours: ",neighbours_data)
print("Info about Neighbours input Variables: \n")
for j in neighbours_data:
    print(drop_data.iloc[j,:])
print("Predicted Target Class:",result)

f) Follow-up with e), what is the predicted probability of fraudulent (i.e., FRAUD = 1)?  If your predicted probability is greater than or equal to your answer in a), then the observation will be classified as fraudulent.  Otherwise, non-fraudulent.  Based on this criterion, will this observation be misclassified?
# In[339]:


probability = fitting_matrix.predict_proba(matrix)
print("Predicting Probabilites based on Train data:",probability)
pred_prob = fitting_matrix.predict_proba(transformation_focal)
print("Predicted Probabilites: ",pred_prob)


# In[338]:





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




