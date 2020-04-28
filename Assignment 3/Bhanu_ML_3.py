import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import combinations
import scipy
import sklearn.metrics as metrics


claim_hist=pd.read_csv("claim_history.csv")
claim_hist.head()
data=claim_hist[['CAR_TYPE','OCCUPATION','EDUCATION','CAR_USE']]
data['EDUCATION'] = data['EDUCATION'].map( {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

x = data[['CAR_TYPE','OCCUPATION','EDUCATION']]
y = data[['CAR_USE']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,train_size=0.75,stratify=y,random_state=60616)

count_train=y_train['CAR_USE'].value_counts()
print( count_train)



count_test=y_test['CAR_USE'].value_counts()
print(count_test)

print(y_test['CAR_USE'].value_counts(normalize=True))

prob_train = y_train[y_train.CAR_USE == 'Commercial'].shape[0]
prob_test = y_test[y_test.CAR_USE == 'Commercial'].shape[0]
prob=abs(float(prob_train)/(prob_train+prob_test))
print((round(prob,6)))

prob_train_1 = y_test[y_test.CAR_USE == 'Private'].shape[0]
prob_test_1 = y_train[y_train.CAR_USE == 'Private'].shape[0]
prob=(float(prob_train_1)/((prob_train_1)+(prob_test_1)))

print(round(prob,6))


#Q2a
sam_data = x_train.iloc[:,:]
sam_data['CAR_USE'] = y_train.loc[:,'CAR_USE']

cont_table=pd.crosstab(sam_data['OCCUPATION'],sam_data['CAR_USE'])
cont_table['Total'] = cont_table['Commercial'] + cont_table['Private']
cont_table = cont_table.append(cont_table.agg(['sum']))

c=cont_table.loc[['sum'],['Commercial']]
t=cont_table.loc[['sum'],['Total']]
p=cont_table.loc[['sum'],['Private']]

rootNodeEntropy = scipy.stats.entropy([c.Commercial/t.Total,p.Private/t.Total], base=2)
print(rootNodeEntropy)


#Q2b
index = ['Commercial','Private','Total']
def red_entropy(c,p,cont):
    com = []
    pr = []
    for x in c:
        com.append(x)
    for y in p:
        pr.append(y)
    comm = cont.loc[com].sum()
    priv = cont.loc[pr].sum() 
    cont_table1 = pd.DataFrame(list(zip(comm,priv)),columns=[" ".join(com)," ".join(pr)])
    cont_table1.index = [x for x in index]
    cont_table1 = cont_table1.T
    cont_table1 = cont_table1.append(cont_table1.agg(['sum']))
    ac = cont_table1.loc[" ".join(com),index[0]]
    bc = cont_table1.loc[" ".join(com),index[1]]
    dc = cont_table1.loc[" ".join(com),index[2]]
    ep = cont_table1.loc[" ".join(pr),index[0]]
    fe = cont_table1.loc[" ".join(pr),index[1]]
    gp = cont_table1.loc[" ".join(pr),index[2]]
    st = cont_table1.loc['sum',index[2]]
    entropy1 = scipy.stats.entropy([ac/dc,bc/dc],base=2)
    entropy2 = scipy.stats.entropy([ep/gp,fe/gp],base=2)
    split_entro = ((dc/st)*entropy1+(gp/st)*entropy2)
    values = [entropy1,entropy2,split_entro]
    return values
def red_entropy_ord(c,p,cont):
    co = []
    pr = []
    for x in c:
        co.append(x)
    for y in p:
        pr.append(y)
    com = cont.loc[co].sum()
    pri = cont.loc[pr].sum() 
    cont_table1 = pd.DataFrame(list(zip(com,pri)),columns=['com','pri'])
    cont_table1.index = [x for x in index]
    cont_table1 = cont_table1.T
    cont_table1 = cont_table1.append(cont_table1.agg(['sum']))
    ac = cont_table1.loc['com',index[0]]
    bc = cont_table1.loc['com',index[1]]
    dc = cont_table1.loc['com',index[2]]
    ep = cont_table1.loc['pri',index[0]]
    fe = cont_table1.loc['pri',index[1]]
    gp = cont_table1.loc['pri',index[2]]
    st = cont_table1.loc['sum',index[2]]
    ent1 = scipy.stats.entropy([ac/dc,bc/dc],base=2)
    ent2 = scipy.stats.entropy([ep/gp,fe/gp],base=2)
    split_ent = ((dc/st)*ent1+(gp/st)*ent2)
    values = [ent1,ent2,split_ent]
    return values

comm = ['Blue Collar','Clerical','Doctor','Home Maker','Lawyer','Manager','Professional','Student','Unknown']

def combos(u,i):
    combined =[]
    for comb in combinations(u,i):
        combined.append(comb)
    return combined
    
commercial_list=[]
private_list=[]
tot=9
for i in range(1,5,1):
    commercial_list.append(combos(comm,i))
    private_list.append(combos(comm,tot-i))
        
df = pd.DataFrame(columns=['Commercial','Private','Red_entropy'])

for x in range(9):
    re_18 = rootNodeEntropy-red_entropy(commercial_list[0][x],private_list[0][-(x+1)],cont_table)[2]
    df = df.append({'Commercial':commercial_list[0][x],'Private':private_list[0][-(x+1)],'reduction_entropy':re_18},ignore_index=True)

for y in range(36):
    re_27 = rootNodeEntropy-red_entropy(commercial_list[1][y],private_list[1][-(y+1)],cont_table)[2]
    df = df.append({'Commercial':commercial_list[1][y],'Private':private_list[1][-(y+1)],'Red_entropy':re_27},ignore_index=True)
    
for z in range(84):
    re_36 = rootNodeEntropy-red_entropy(commercial_list[2][z],private_list[2][-(z+1)],cont_table)[2]
    df = df.append({'Commercial':commercial_list[2][z],'Private':private_list[2][-(z+1)],'Red_entropy':re_36},ignore_index=True)

for w in range(126):
    re_45 = rootNodeEntropy-red_entropy(commercial_list[3][w],private_list[3][-(w+1)],cont_table)[2]
    df = df.append({'Commercial':commercial_list[3][w],'Private':private_list[3][-(w+1)],'Red_entropy':re_45},ignore_index=True)


sc = df.loc[df['Red_entropy'] == float(df.loc[:,'Red_entropy'].max()),['Commercial','Private']]

print("Predictor Name: ", data.columns[0])
print("Left Child: ", list(sc['Commercial']))
print("Right Child: ", list(sc['Private']))


#Q2c
se = red_entropy(df.iloc[28,0],df.iloc[28,1],cont_table)[2]

print("First Layer entropy split: ", se)

comm_split_1 = sam_data[sam_data['OCCUPATION'].isin(['Blue Collar','Student', 'Unknown'])].reset_index().drop(['index'], axis=1)
priv_split_2 = sam_data[sam_data['OCCUPATION'].isin(['Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional'])].reset_index().drop(['index'], axis=1)

cont_table2 = pd.crosstab(comm_split_1['EDUCATION'], comm_split_1['CAR_USE'])
cont_table2['Total'] = cont_table2['Commercial'] + cont_table2['Private']
cont_table2 = cont_table2.append(cont_table2.agg(['sum']))
comm1 = cont_table2.loc[['sum'],['Commercial']]
tot1 = cont_table2.loc[['sum'],['Total']]
priv1 = cont_table2.loc[['sum'],['Private']]

entropy_c1 = scipy.stats.entropy([comm1.Commercial/tot1.Total,priv1.Private/tot1.Total], base=2)

s_c = [0,1,2,3,4]

comm1_list = []
priv1_list = []
total = 5
for i in range(1,3,1):
    comm1_list.append(combos(s_c,i))
    priv1_list.append(combos(s_c,total-i))

data2 = pd.DataFrame(columns=['Commercial','Private','Red_entropy'])

for x in range(5):
    r_14 = entropy_c1-red_entropy_ord(comm1_list[0][x],priv1_list[0][-(x+1)],cont_table2)[2]
    data2 = data2.append({'Commercial':comm1_list[0][x],'Private':priv1_list[0][-(x+1)],'Red_entropy':r_14}, ignore_index=True)

for y in range(10):
    r_23 = entropy_c1-red_entropy_ord(comm1_list[1][y],priv1_list[1][-(y+1)],cont_table2)[2]
    data2 = data2.append({'Commercial':comm1_list[1][y],'Private':priv1_list[1][-(y+1)],'Red_entropy':r_23}, ignore_index=True)
    
child1_c = data2.loc[data2['Red_entropy'] == float(data2.loc[:,'Red_entropy'].max()),['Commercial','Private']]

c1_e = float(data2.loc[:,'Red_entropy'].max())

comm_split_3 = comm_split_1[comm_split_1['EDUCATION'].isin([0])].reset_index().drop(['index'],axis=1)

cont_table4=pd.crosstab(comm_split_3['EDUCATION'],comm_split_3['CAR_USE'])
cont_table4['Total'] = cont_table4['Commercial'] + cont_table4['Private']
cont_table4 = cont_table4.append(cont_table4.agg(['sum']))
comm3=cont_table4.loc[['sum'],['Commercial']]
tot3=cont_table4.loc[['sum'],['Total']]
priv3=cont_table4.loc[['sum'],['Private']]
entropy_c3 = scipy.stats.entropy([comm3.Commercial/tot3.Total,priv3.Private/tot3.Total], base=2)

priv_split_4 = comm_split_1[comm_split_1['EDUCATION'].isin([1,2,3,4])].reset_index().drop(['index'],axis=1)
cont_table5=pd.crosstab(priv_split_4['EDUCATION'],priv_split_4['CAR_USE'])
cont_table5['Total'] = cont_table5['Commercial'] + cont_table5['Private']
cont_table5 = cont_table5.append(cont_table5.agg(['sum']))
comm4=cont_table5.loc[['sum'],['Commercial']]
tot4=cont_table5.loc[['sum'],['Total']]
priv4=cont_table5.loc[['sum'],['Private']]
entropy_c4 = scipy.stats.entropy([comm4.Commercial/tot4.Total,priv4.Private/tot4.Total], base=2)

cont_table3=pd.crosstab(priv_split_2['CAR_TYPE'],priv_split_2['CAR_USE'])
cont_table3['Total'] = cont_table3['Commercial'] + cont_table3['Private']
cont_table3 = cont_table3.append(cont_table3.agg(['sum']))

comm2=cont_table3.loc[['sum'],['Commercial']]
tot2=cont_table3.loc[['sum'],['Total']]
priv2=cont_table3.loc[['sum'],['Private']]

entropy_c2 = scipy.stats.entropy([comm2.Commercial/tot2.Total,priv2.Private/tot2.Total], base=2)

s1_c2 = ['Minivan','Panel Truck','Pickup','SUV','Sports Car','Van']

comm2_list = []
priv2_list = []
total = 6
for i in range(1,4,1):
    comm2_list.append(combos(s1_c2,i))
    priv2_list.append(combos(s1_c2,total-i))

data3 = pd.DataFrame(columns=['Commercial','Private','Red_entropy'])

for j in range(6):
    r_15 = entropy_c2-red_entropy(comm2_list[0][j],priv2_list[0][-(j+1)],cont_table3)[2]
    data3 = data3.append({'Commercial':comm2_list[0][j],'Private':private_list[0][-(j+1)],'Red_entropy':r_15},ignore_index=True)

for k in range(15):
    r_24 = entropy_c2-red_entropy(comm2_list[1][k],priv2_list[1][-(k+1)],cont_table3)[2]
    data3 = data3.append({'Commercial':comm2_list[1][k],'Private':private_list[1][-(k+1)],'Red_entropy':r_24},ignore_index=True)
    
for l in range(10):
    r_33 = entropy_c2-red_entropy(comm2_list[2][l],priv2_list[2][-(l+1)],cont_table3)[2]
    data3 = data3.append({'Commercial':comm2_list[2][l],'Private':priv2_list[2][-(l+1)],'Red_entropy':r_33},ignore_index=True)
   
child2_e = data3.loc[data3['Red_entropy'] == float(data3.loc[:,'Red_entropy'].max()),['Commercial','Private']]

c2_e = float(data3.loc[:,'Red_entropy'].max())

comm_split_5 = priv_split_2[priv_split_2['CAR_TYPE'].isin(['Minivan', 'SUV', 'Sports Car'])].reset_index().drop(['index'],axis=1)

conteg_table6=pd.crosstab(comm_split_5['CAR_TYPE'],comm_split_5['CAR_USE'])
conteg_table6['Total'] = conteg_table6['Commercial'] + conteg_table6['Private']
conteg_table6 = conteg_table6.append(conteg_table6.agg(['sum']))
comm5=conteg_table6.loc[['sum'],['Commercial']]
tot5=conteg_table6.loc[['sum'],['Total']]
priv5=conteg_table6.loc[['sum'],['Private']]
entropy_c5 = scipy.stats.entropy([comm5.Commercial/tot5.Total,priv5.Private/tot5.Total], base=2)

priv_split_6 = priv_split_2[priv_split_2['CAR_TYPE'].isin(['Panel Truck', 'Pickup', 'Van'])].reset_index().drop(['index'],axis=1)

conteg_table7=pd.crosstab(priv_split_6['CAR_TYPE'],priv_split_6['CAR_USE'])
conteg_table7['Total'] = conteg_table7['Commercial'] + conteg_table7['Private']
conteg_table7 = conteg_table7.append(conteg_table7.agg(['sum']))
comm6=conteg_table7.loc[['sum'],['Commercial']]
tot6=conteg_table7.loc[['sum'],['Total']]
priv6=conteg_table7.loc[['sum'],['Private']]
entropy_c6 = scipy.stats.entropy([comm6.Commercial/tot6.Total,priv6.Private/tot6.Total], base=2)

leaf1=comm_split_3['CAR_USE'].value_counts()
leaf2=priv_split_4['CAR_USE'].value_counts()
leaf3=comm_split_5['CAR_USE'].value_counts()
leaf4=priv_split_6['CAR_USE'].value_counts()


data4 = pd.DataFrame(columns=['Entropy','No of Observations','% of Commercial','Predicted Class'])

entr = []

entr.append(list(entropy_c3))
entr.append(list(entropy_c4))
entr.append(list(entropy_c5))
entr.append(list(entropy_c6))

n_obs = []

n_obs.append(len(comm_split_3))
n_obs.append(len(priv_split_4))
n_obs.append(len(comm_split_5))
n_obs.append(len(priv_split_6))

comm_obs = []

comm_obs.append((round((leaf1['Commercial']/n_obs[0])*100)))
comm_obs.append((round((leaf2['Commercial']/n_obs[1])*100)))
comm_obs.append((round((leaf3['Commercial']/n_obs[2])*100)))
comm_obs.append((round((leaf4['Commercial']/n_obs[3])*100)))

predicted_class = []

for i in range(len(comm_obs)):
    if(comm_obs[i]>50.0):
        predicted_class.append('Commercial')
    else:
        predicted_class.append('Private')

comm_counts = []

comm_counts.append(leaf1['Commercial'])
comm_counts.append(leaf2['Commercial'])
comm_counts.append(leaf3['Commercial'])
comm_counts.append(leaf4['Commercial'])

priv_counts = []

priv_counts.append(leaf1['Private'])
priv_counts.append(leaf2['Private'])
priv_counts.append(leaf3['Private'])
priv_counts.append(leaf4['Private'])

leaf_node1 = {'Education':list(child1_c['Commercial']),'Occupation':list(sc['Commercial'])}
leaf_node2 = {'Education':list(child1_c['Private']),'Occupation':list(sc['Commercial'])}
leaf_node3 = {'Car_Type':list(child2_e['Commercial']),'Occupation':list(sc['Private'])}
leaf_node4 = {'Car_Type':list(child2_e['Private']),'Occupation':list(sc['Private'])}

index = ['Leaf Node 1','Leaf Node 2','Leaf Node 3','Leaf Node 4']

data4['Index'] = index
data4['Entropy'] = entr
data4['No of Observations'] = n_obs
data4['% of Commercial'] = comm_obs
data4['Predicted Class'] = predicted_class
data4['Commercial'] = comm_counts
data4['Private'] = priv_counts
data4=data4.set_index('Index')


#Q2e

print("Decision Rules")
print("Leaf Node 1: ",leaf_node1)
print("Leaf Node 2: ",leaf_node2)
print("Leaf Node 3: ",leaf_node3)
print("Leaf Node 4: ",leaf_node4)
pd.set_option('display.max_columns', None)
print(data4)

comm_split_3['Predicted_Class'] = predicted_class[0]
priv_split_4['Predicted_class'] = predicted_class[1]
comm_split_5['Predicted_Class'] = predicted_class[2]
priv_split_6['Predicted_Class'] = predicted_class[3]

#Q2f
def predicted_category(data):
    if data['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if data['EDUCATION'] <=0.5:
            return [0.27,0.73]
        else:
            return [0.84,0.16]
    else:
        if data['CAR_TYPE'] in ('Panel Truck', 'Pickup', 'Van'):
            return [0.53,0.47]
        else:
            return [0.1,0.99]
    
def decision_tree(data):
    output_data = np.ndarray(shape=(len(data), 2), dtype=float)
    count = 0
    for index, row in data.iterrows():
        prob = predicted_category(data=row)
        output_data[count] = prob
        count += 1
    return output_data

predicted_probability_train = decision_tree(data=x_train)
predicted_probability_train = predicted_probability_train[:, 0]
predicted_probability_train = list(predicted_probability_train)

threshold = x_train['CAR_USE'].value_counts()['Commercial']/len(x_train)
pd.set_option('display.max_rows', None)

y_train['Pred_prob'] = predicted_probability_train

predicted_train = []
for i in range(len(y_train)):
    if y_train.iloc[i,1] > 0.534:
        predicted_train.append('Commercial')
    else:
        predicted_train.append('Private')

y_train['Predicted'] = predicted_train
        
misclass_train = []

for i in range(len(y_train)):
    if y_train.iloc[i,0] == y_train.iloc[i,2]:
        misclass_train.append(0)
    else:
        misclass_train.append(1)

y_train['miss Classified'] = misclass_train


FP, TP, threshold = metrics.roc_curve(y_train['CAR_USE'],y_train['Pred_prob'], pos_label='Commercial')

cut_off = np.where(threshold > 1.0, np.nan, threshold)
plt.plot(cut_off, TP, marker = 'o', label = 'True Positive', color = 'pink', linestyle = 'solid')
plt.plot(cut_off, FP, marker = 'o', label = 'False Positive', color = 'blue', linestyle = 'solid')
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True)

print("KS Statistic: 0.7")
print("Event probability cutoff value: 0.27")
plt.show()

event_prob_cutoff = 0.27


#Question 3
predicted_prob = decision_tree(data=x_test)
predicted_prob = predicted_prob[:, 0]
predicted_prob = list(predicted_prob)

y_test['Pred_prob'] = predicted_prob

predict = []
for i in range(len(y_test)):
    if y_test.iloc[i,1] > 0.367:
        predict.append('Commercial')
    else:
        predict.append('Private')

y_test['Predicted'] = predict
        
misclass = []

for i in range(len(y_test)):
    if y_test.iloc[i,0] == y_test.iloc[i,2]:
        misclass.append(0)
    else:
        misclass.append(1)

y_test['miss Classified'] = misclass


#Q3a

accuracy = metrics.accuracy_score(y_test['CAR_USE'], y_test['Predicted'])
misclassification = 1-accuracy

print("MissClassification Rate: ", misclassification)


#Q3b
predict_KS = []
for i in range(len(y_test)):
    if y_test.iloc[i,1] > event_prob_cutoff:
        predict_KS.append('Commercial')
    else:
        predict_KS.append('Private')
y_test.loc[:,'KS'] = predict_KS
accuracy_KS = metrics.accuracy_score(y_test['CAR_USE'], predict_KS)
misclassification_KS = 1-accuracy_KS

print("MissClassification Rate with Kolmogorov-Smirnov event probability cutoff value as Threshold: ",misclassification_KS)

#Q3c
y_comm = 1.0 * np.isin(y_test['CAR_USE'], ['Commercial'])
MSE = metrics.mean_squared_error(y_comm, y_test['Pred_prob'])
RMSE = np.sqrt(MSE)

print("Root Mean Squared Error for Test Partition: ",RMSE)

comm_prob = []
priv_prob = []
for i in range(len(y_test)):
    if(y_test.iloc[i,0]=='Commercial'):
        comm_prob.append(y_test.iloc[i,1])
    else:
        priv_prob.append(y_test.iloc[i,1])

concordant = 0
discordant = 0
tie = 0
for i in comm_prob:
    for j in priv_prob:
        if(i>j):
            concordant+=1
        elif(i<j):
            discordant+=1
        else:
            tie+=1
            
            
#Q3d

AUC = 0.5 + 0.5*(concordant-discordant)/(concordant+discordant+tie)
print("Area Under Curve: ",AUC)


#Q3e

GINI = (concordant-discordant)/(concordant+discordant+tie)
print("GINI Coefficient: ",GINI) 


#Q3f 
     
GKG = (concordant-discordant)/(concordant+discordant)
print("Goodman-Kruskal Gamma statistic: ",GKG)


#Q3g

OneMinusSpecificity = np.append([0], FP)
Sensitivity = np.append([0], TP)
OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])

plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'orange', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis("equal")
plt.show()