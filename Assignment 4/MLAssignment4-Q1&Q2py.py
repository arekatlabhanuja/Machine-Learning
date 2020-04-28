#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas
import seaborn as s
import sympy
import matplotlib.pyplot as plt
import scipy
import numpy as np
import statsmodels.api as stats
import sklearn.ensemble as ensemble

1) a)List the aliased columns that you found in your model matrix.
# In[2]:


def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)


# In[4]:


def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pandas.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pandas.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)  ## Need to work on this more

    # Return model statistics
    return (thisLLK, thisDF, fullParams)


# In[5]:


purchase_likelihood=pandas.read_csv('Downloads/Purchase_Likelihood.csv',delimiter=',' , usecols = ['group_size', 'homeowner', 'married_couple','insurance'])
purchase_likelihood=purchase_likelihood.dropna()


# In[17]:


no_objs=purchase_likelihood.shape[0]
purchase_likelihood.head()
y=purchase_likelihood['insurance'].astype('category')
x_group=pandas.get_dummies(purchase_likelihood[['group_size']].astype('category'))
x_home=pandas.get_dummies(purchase_likelihood[['homeowner']].astype('category'))
x_marcou=pandas.get_dummies(purchase_likelihood[['married_couple']].astype('category'))
#intercept only model
#reference from professor's code
design_X = pandas.DataFrame(y.where(y.isnull(), 1))
lik0, df0, fullParams0 = build_mnlogit (design_X, y, debug = 'Y')
print("Number of free parameters: ",df0)
print("Log Likelihood: ",lik0)


# In[6]:


no_objs=purchase_likelihood.shape[0]
purchase_likelihood.head()
y=purchase_likelihood['insurance'].astype('category')
x_group=pandas.get_dummies(purchase_likelihood[['group_size']].astype('category'))
x_home=pandas.get_dummies(purchase_likelihood[['homeowner']].astype('category'))
x_marcou=pandas.get_dummies(purchase_likelihood[['married_couple']].astype('category'))


# In[22]:


#intercept + group_size
design_X = stats.add_constant(x_group, prepend=True)
LLK_1G, DF_1G, fullParams_1G = build_mnlogit (design_X, y, debug = 'N')
testDev = 2 * (LLK_1G - lik0)
testDF = DF_1G - df0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print("Number of free parameters: ",DF_1G)
print("Log Likelihood: ",LLK_1G)
print('Deviance = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print("Feature Importance index: ",round(-np.log10(testPValue),4))


# In[24]:


#intercept + group_size + home_owner
designX = x_group
designX = designX.join(x_home)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H, DF_1G_1H, fullParams_1G_1H = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H - LLK_1G)
testDF = DF_1G_1H - DF_1G
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print("Number of free parameters: ",DF_1G_1H)
print("Log Likelihood: ",LLK_1G_1H)
print('Deviance = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print("Feature Importance index: ",round(-np.log10(testPValue),4))


# In[25]:


#intercept + group_size + home_owner + married_couple
designX = x_group
designX = designX.join(x_home)
designX=designX.join(x_marcou)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M, DF_1G_1H_1M, fullParams_1G_1H = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M - LLK_1G_1H)
testDF = DF_1G_1H_1M - DF_1G_1H
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print("Number of free parameters: ",DF_1G_1H_1M)
print("Log Likelihood: ",LLK_1G_1H_1M)
print('Deviance = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print("Feature Importance index: ",round(-np.log10(testPValue),4))


# In[26]:


#intercept + group_size + home_owner + married_couple + group_size * homeowner
designX = x_group
designX = designX.join(x_home)
designX = designX.join(x_marcou)

# Create the columns for the group_size * homeowner interaction effect
xGH = create_interaction(x_group, x_home)
designX = designX.join(xGH)

designX = stats.add_constant(designX, prepend=True)
LLK_2GH, DF_2GH, fullParams_2GH = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_2GH - LLK_1G_1H_1M) 
testDF = DF_2GH - DF_1G_1H_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print("Number of free parameters: ",DF_2GH)
print("Log Likelihood: ",LLK_2GH)
print('Deviance = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print("Feature Importance index: ",round(-np.log10(testPValue),4))


# In[27]:


#intercept + group_size + home_owner + married_couple + group_size * homeowner + group_size * married_couple
designX = x_group
designX = designX.join(x_home)
designX = designX.join(x_marcou)

# Create the columns for the group_size * homeowner interaction effect
xGH = create_interaction(x_group, x_home)
designX = designX.join(xGH)
designX = stats.add_constant(designX, prepend=True)

#Create the columns for the group_size * married_couple interaction effect
xGM = create_interaction(x_group, x_marcou)
designX = designX.join(xGM)
designX = stats.add_constant(designX, prepend=True)

LLK_2GH_2GM, DF_2GH_2GM, fullParams_2GH_2GM = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_2GH_2GM - LLK_2GH)
testDF = DF_2GH_2GM- DF_2GH
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print("Number of free parameters: ",DF_2GH_2GM)
print("Log Likelihood: ",LLK_2GH_2GM)
print('Deviance = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print("Feature Importance index: ",round(-np.log10(testPValue),4))


# In[28]:


#intercept + group_size + home_owner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple
designX = x_group
designX = designX.join(x_home)
designX = designX.join(x_marcou)

# Create the columns for the group_size * homeowner interaction effect
xGH = create_interaction(x_group, x_home)
designX = designX.join(xGH)
designX = stats.add_constant(designX, prepend=True)

#Create the columns for the group_size * married_couple interaction effect
xGM = create_interaction(x_group, x_marcou)
designX = designX.join(xGM)
designX = stats.add_constant(designX, prepend=True)

#Create the columns for the homeowner * married_couple interaction effect
xHM = create_interaction(x_home, x_marcou)
designX = designX.join(xHM)
designX=stats.add_constant(designX, prepend=True)

LLK_2GH_2GM_2HM, DF_2GH_2GM_2HM, fullParams_2GH_2GM_2HM = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2GH_2GM_2HM - LLK_2GH_2GM)
testDF = DF_2GH_2GM_2HM- DF_2GH_2GM
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print("Number of free parameters: ",DF_2GH_2GM_2HM)
print("Log Likelihood: ",LLK_2GH_2GM_2HM)
print('Deviance = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print("Feature Importance index: ",round(-np.log10(testPValue),4))


# In[34]:


print("Aliased columns in the model matrix are:")
print("group_size_4")
print("homeowner_1")
print("married_couple_1")
print("group_size _1* homeowner_1")
print("group_size_2 * homeowner_1")
print("group_size_3 * homeowner_1")
print("group_size_4 * homeowner_0")
print("group_size_4 * homeowner_1")
print("group_size_1 * married_couple_1")
print("group_size_2 * married_couple_1")
print("group_size_3 * married_couple_1")
print("group_size_4 * married_couple_1")
print("group_size_4 * married_couple_0")
print("homeowner_0 * married_couple_1")
print("homeowner_1 * married_couple_0")
print("homeowner_1 * married_couple_1")

2) a) For each of the sixteen possible value combinations of the three features, calculate the predicted probabilities for insurance = 0, 1, 2 based on your multinomial logistic model.  
# In[11]:


designX = x_group
designX = designX.join(x_home)
designX = designX.join(x_marcou)

# Create the columns for the group_size * homeowner interaction effect
xGH = create_interaction(x_group, x_home)
designX = designX.join(xGH)
designX = stats.add_constant(designX, prepend=True)

#Create the columns for the group_size * married_couple interaction effect
xGM = create_interaction(x_group, x_marcou)
designX = designX.join(xGM)
designX = stats.add_constant(designX, prepend=True)

#Create the columns for the homeowner * married_couple interaction effect
xHM = create_interaction(x_home, x_marcou)
designX = designX.join(xHM)
designX=stats.add_constant(designX, prepend=True)

model = stats.MNLogit(y,designX)
fit = model.fit(method='newton',full_output=True,maxiter=100,tol=1e-8)

gs = [1,2,3,4]
ho = [0,1]
mc = [0,1]

data = [ ]

for g in gs:
    for h in ho:
        for m in mc:
            d = [g,h,m]
            data = data+[d]
            
in_df = pandas.DataFrame(data,columns=['group_size','homeowner','married_couple'])
x_group = pandas.get_dummies(in_df[['group_size']].astype('category'))
x_home = pandas.get_dummies(in_df[['homeowner']].astype('category'))
x_marcou = pandas.get_dummies(in_df[['married_couple']].astype('category'))
designX = x_group
designX = designX.join(x_home)
designX = designX.join(x_marcou)

# Create the columns for the group_size * homeowner interaction effect
xGH = create_interaction(x_group, x_home)
designX = designX.join(xGH)
designX = stats.add_constant(designX, prepend=True)

#Create the columns for the group_size * married_couple interaction effect
xGM = create_interaction(x_group, x_marcou)
designX = designX.join(xGM)
designX = stats.add_constant(designX, prepend=True)

#Create the columns for the homeowner * married_couple interaction effect
xHM = create_interaction(x_home, x_marcou)
designX = designX.join(xHM)
designX=stats.add_constant(designX, prepend=True)

predict = fit.predict(exog=designX)

predict['i0']= predict[0]
predict['i1']= predict[1]
predict['i2']= predict[2]

prob_df = pandas.concat([in_df,predict[['i0','i1','i2']]],axis=1)
print(prob_df)


# In[22]:


prob_df

b)	(5 points) Based on your answers in (a), what value combination of group_size, homeowner, and married_couple will maximize the odds value Prob(insurance = 1) / Prob(insurance = 0)?  What is that maximum odd value?
# In[19]:


lis = []
for i in range(len(prob_df)):
    lis.append(prob_df.iloc[i,4]/prob_df.iloc[i,3])
m = lis.index(max(lis))
max_val = max(lis)
print('Maximum value of insurance=1/insurance=0 is',max_val)
print('group_size:',prob_df.iloc[m,0])
print('homeowner:',prob_df.iloc[m,1])
print('married_couple:',prob_df.iloc[m,2])

c)	(5 points) Based on your model, what is the odds ratio for group_size = 3 versus group_size = 1, and insurance = 2 versus insurance = 0?
# In[35]:


odds_g3_i2 = (purchase_likelihood[purchase_likelihood['group_size']==3].groupby('insurance').size()[2]/purchase_likelihood[purchase_likelihood['group_size']==3].shape[0]) 
odds_g3_i0 = (purchase_likelihood[purchase_likelihood['group_size']==3].groupby('insurance').size()[0]/purchase_likelihood[purchase_likelihood['group_size']==3].shape[0])
odds1 = odds_g3_i2/odds_g3_i0
odds_g1_i2 = (purchase_likelihood[purchase_likelihood['group_size']==1].groupby('insurance').size()[2]/purchase_likelihood[purchase_likelihood['group_size']==1].shape[0]) 
odds_g2_i0 = (purchase_likelihood[purchase_likelihood['group_size']==1].groupby('insurance').size()[0]/purchase_likelihood[purchase_likelihood['group_size']==1].shape[0])
odds2 = odds_g1_i2/odds_g2_i0
o = odds1/odds2
print("Required Odds Ratio: ",(o))

d)	(5 points) Based on your model, what is the odds ratio for homeowner = 1 versus homeowner = 0, and insurance = 0 versus insurance = 1?
# In[24]:


odds_h1_i0 = (purchase_likelihood[purchase_likelihood['homeowner']==1].groupby('insurance').size()[0]/purchase_likelihood[purchase_likelihood['homeowner']==1].shape[0]) 
odds_h1_i1 = (purchase_likelihood[purchase_likelihood['homeowner']==1].groupby('insurance').size()[1]/purchase_likelihood[purchase_likelihood['homeowner']==1].shape[0])
odds1 = odds_h1_i0/odds_h1_i1
odds_h0_i0 = (purchase_likelihood[purchase_likelihood['homeowner']==0].groupby('insurance').size()[0]/purchase_likelihood[purchase_likelihood['homeowner']==0].shape[0]) 
odds_h0_i1 = (purchase_likelihood[purchase_likelihood['homeowner']==0].groupby('insurance').size()[1]/purchase_likelihood[purchase_likelihood['homeowner']==0].shape[0])
odds2 = odds_h0_i0/odds_h0_i1
oddsR = odds1/odds2
print("Required Odds Ratio: ",oddsR)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




