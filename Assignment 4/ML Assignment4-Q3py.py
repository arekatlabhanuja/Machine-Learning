#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import scipy
from sklearn import naive_bayes
import statsmodels.api as stats

3) a)(5 points) Show in a table the frequency counts and the Class Probabilities of the target variable.
# In[2]:


feature = ['group_size','homeowner','married_couple']
target = 'insurance'
purchase_likelihood = pandas.read_csv('C:\\Users\\barek\\Downloads\\Purchase_Likelihood.csv',
                              delimiter=',' ,
                              usecols = feature + [target])
purchase_likelihood = purchase_likelihood.dropna()
frequency=purchase_likelihood.groupby('insurance').size()
tab=pandas.DataFrame(columns = ['Frequency Count','Class Probability'])
tab['Frequency Count']=frequency
tab['Class Probability']=tab['Frequency Count']/purchase_likelihood.shape[0]
print(tab)


# b) Show the crosstabulation table of the target variable by the feature group_size.  The table contains the frequency counts.

# In[3]:


counts = pandas.crosstab(purchase_likelihood.insurance, purchase_likelihood.group_size, margins = False, dropna = False)
print("The crosstabulation table of the target variable by the feature group_size:")
display(counts)


# c) Show the crosstabulation table of the target variable by the feature homeowner.  The table contains the frequency counts.

# In[5]:


counts = pandas.crosstab(purchase_likelihood.insurance, purchase_likelihood.homeowner, margins = False, dropna = False)
print("The crosstabulation table of the target variable by the feature homeowner:")
display(counts)


# d) Show the crosstabulation table of the target variable by the feature married_couple.  The table contains the frequency counts.

# In[6]:


counts = pandas.crosstab(purchase_likelihood.insurance, purchase_likelihood.married_couple, margins = False, dropna = False)
print("The crosstabulation table of the target variable by the feature married_couple:")
display(counts)

e) Calculate the Cramer’s V statistics for the above three crosstabulations tables.  Based on these Cramer’s V statistics, which feature has the largest association with the target insurance?
# In[7]:


#Ref from class notes
def ChiSquareTest (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):
    obsCount = pandas.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))
       
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
    chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = numpy.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)


# In[8]:


cramerV_group_size = ChiSquareTest(purchase_likelihood.group_size, purchase_likelihood.insurance)[3]
print("Cramer's V Value of group size is: ", cramerV_group_size)
cramerV_homeowner = ChiSquareTest(purchase_likelihood.homeowner, purchase_likelihood.insurance)[3]
print("Cramer's V Value of Homeowner is: ", cramerV_homeowner)
cramerV_married_couple = ChiSquareTest(purchase_likelihood.married_couple, purchase_likelihood.insurance)[3]
print("Cramer's V Value of Married_couple is: ", cramerV_married_couple)


# In[9]:


max_feature=max(cramerV_group_size,cramerV_homeowner,cramerV_married_couple)
print("The feature which has largest assocaition with the target insurance is homeowner with",max_feature)

f) For each of the sixteen possible value combinations of the three features, calculate the predicted probabilities for insurance = 0, 1, 2 based on the Naïve Bayes model.  List your answers in a table with proper labeling.
# In[15]:


cat_column = ['group_size', 'homeowner', 'married_couple']
target_col = []
x_Train = purchase_likelihood[cat_column].astype('category')
y_Train = purchase_likelihood['insurance'].astype('category')

naivebayes = naive_bayes.MultinomialNB(alpha = 1.0e-10)
Fit_data = naivebayes.fit(x_Train, y_Train)
group_size_data = [1,2,3,4]
home_owner_data = [0,1]
married_couple_data = [0,1]
insurance_data = [0,1,2]

final_data = []

for group in group_size_data:
    for home in home_owner_data:
        for married_couple in married_couple_data:
            data = [group,home,married_couple]
            final_data = final_data + [data]

x_col = pandas.DataFrame(final_data, columns=['group_size','homeowner','married_couple'])
x_col = x_col[cat_column].astype('category')
y_predprob = pandas.DataFrame(naivebayes.predict_proba(x_col), columns = ['prob(insurance=0)', 'prob(insurance=1)','prob(insurance=2)'])
y_score = pandas.concat([x_col, y_predprob], axis = 1)
                                                                                      
print(y_score)

g)Based on your model, what value combination of group_size, homeowner, and married_couple will maximize the odds value Prob(insurance = 1) / Prob(insurance = 0)?  What is that maximum odd value?
# In[14]:


y_score['odd value(prob(insurance=1)/prob(insurance=0))'] = y_score['prob(insurance=1)'] / y_score['prob(insurance=0)']
print(y_score[['group_size','homeowner','married_couple','odd value(prob(insurance=1)/prob(insurance=0))']])
print(y_score.loc[y_score['odd value(prob(insurance=1)/prob(insurance=0))'].idxmax()])


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




