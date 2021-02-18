# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:05:53 2021

@author: Home
"""

import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.utils import resample
import numpy as np
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.metrics import f1_score

dataset = pd.read_csv("assessmentFile.csv")


dataset = dataset.drop(["isFlaggedFraud"],axis=1)


# processing for visualization graphs #

Pie = dataset[['amount','type']]
Pie = Pie.astype({"type": str})
Pie = Pie.groupby('type')
Pie = pd.DataFrame({'sum' : Pie['amount'].sum()}).reset_index()
Pie = Pie.sort_values(by=['sum'],ascending =False)
Pie = Pie.set_index("type")

Pie['sum'] = 100 * Pie['sum'] / Pie['sum'].sum()
Pie.to_csv("PieChart.csv")


#dataset.amount.sum()  227199221.28
#dataset.amount.sum()

# processing for Line graphs #
Line = dataset[['step','isFraud']]
Line = Line[Line.isFraud == 1]
Line['Hour'] = 0



Line1 = Line[['step','isFraud']]
Line1 = Line1.astype({"step": int})
Line1 = Line1.groupby('step')
Line1 = pd.DataFrame({'count' : Line1['isFraud'].size()}).reset_index()
Line1 = Line1.sort_values(by=['step'],ascending =True)

Line1['Hour']= 0
lis = []
j = 1
for i in range(0,len(Line1['step'])):
    if (j == 25):
        j=1
        Line1['Hour'].iloc[i] = j
    else:
        Line1['Hour'].iloc[i] = j
    j = j + 1

Line1= Line1[['Hour','count']]


Line1 = Line1[['Hour','count']]
Line1 = Line1.astype({"Hour": int})
Line1 = Line1.groupby('Hour')
Line1 = pd.DataFrame({'sum' : Line1['count'].sum()}).reset_index()
Line1 = Line1.sort_values(by=['Hour'],ascending =True)




Line1 = Line1.set_index("Hour")
Line1.to_csv("LineChart.csv")



# Processing for Stacked Graph #
stacked = dataset[['amount','nameDest']]

consumer_to_merchant = stacked[stacked['nameDest'].str.contains("M")]
consumer_to_consumer =  stacked[stacked['nameDest'].str.contains("C")]

value1 = consumer_to_consumer.amount.sum()
value2 = consumer_to_merchant.amount.sum() 

lis = [["Consumer-to-Consumer",value1],["Consumer-to-Merchant",value2]]
stacked1 = pd.DataFrame(lis,columns = ["Type of Transactions", "sum of transaction"])
stacked1 = stacked1.set_index("Type of Transactions")
stacked1 = stacked1.transpose()
stacked1.to_csv("Stacked.csv")



# Processing for Bar chart #

Barchart = dataset[['nameOrig','newbalanceOrig']]

Barchart = Barchart.astype({"nameOrig": str})
Barchart = Barchart.groupby('nameOrig')
Barchart = pd.DataFrame({'newbalanceOrig' : Barchart['newbalanceOrig'].max()}).reset_index()
Barchart = Barchart.sort_values(by=['newbalanceOrig'],ascending =False)
Barchart = Barchart.iloc[0:10]
Barchart = Barchart.set_index("nameOrig")
Barchart.to_csv("Barchart.csv")


# Processing for Table #

Table = dataset[['nameOrig','amount']]
Table = Table.astype({"nameOrig": str})
Table = Table.groupby('nameOrig')
Table = pd.DataFrame({'Sum_of_transactions' : Table['amount'].sum()}).reset_index()
Table = Table.sort_values(by=['Sum_of_transactions'],ascending =False)
Table = Table[Table.Sum_of_transactions > 100000000]
Table = Table.set_index("nameOrig")
Table.to_csv("Table.csv")








# Data mining Algorithms:

df1 = dataset
#dataset = df1
dataset.columns

dataset['step'] = dataset['step'].astype('category').cat.codes
dataset['type'] = dataset['type'].astype('category').cat.codes
dataset['amount'] = dataset['amount'].astype('category').cat.codes
dataset['nameOrig'] = dataset['nameOrig'].astype('category').cat.codes
dataset['oldbalanceOrg'] = dataset['oldbalanceOrg'].astype('category').cat.codes
dataset['newbalanceOrig'] = dataset['newbalanceOrig'].astype('category').cat.codes
dataset['nameDest'] = dataset['nameDest'].astype('category').cat.codes
dataset['oldbalanceDest'] = dataset['oldbalanceDest'].astype('category').cat.codes
dataset['newbalanceDest'] = dataset['newbalanceDest'].astype('category').cat.codes


Class_labels = dataset['isFraud']
dataset = dataset.drop(['isFraud'],axis =1)
Class_labels = Class_labels.astype('category').cat.codes


#chi square test 
chi_scores = chi2(dataset,Class_labels)
chi_scores
p_values = pd.Series(chi_scores[1],index = dataset.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()

#NOW removing newbalanceDest,Type,Step as they are independent according to chi square test

df1 = df1.drop(['newbalanceDest','type','step'],axis=1) 

df1.columns

#upsampling

df_majority = df1[df1.isFraud==0]
df_minority = df1[df1.isFraud==1]



df_minority_upsampled = resample(df_minority,replace=True,n_samples=6346191,random_state=30000)
print(df_minority_upsampled)

df1 = pd.concat([df_majority, df_minority_upsampled])
df1 = shuffle(df1)


class_labels = df1['isFraud']
df1 = df1.drop(['isFraud'],axis=1)
dense = df1.iloc[:9519288]
dense_classlabels = class_labels.iloc[:9519288]

#75% break in test and train

test = df1.iloc[9519288:]
test_classlabels = class_labels.iloc[9519288:]





tre = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
tre = tre.fit(dense,dense_classlabels)
print(tre)
tree.plot_tree(tre)
 

predictions = []
for i in range(0,len(test)):
    
    pre = np.asarray(test.iloc[i])
    output = tre.predict([pre])
    predictions.append(output[0])
print(len(predictions))

from sklearn import metrics

test_labels = np.asarray(test_classlabels)
predictions = np.asarray(predictions)
# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
print(metrics.confusion_matrix(test_labels, predictions, labels=[1, 0]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(test_labels, predictions, labels=[1,0]))



f1_score(test_classlabels, predictions, average='micro')
with open('test_outputHW2CS584.txt', 'w') as f:
    for item in predictions:
        f.write("%s\n" % item)




# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
        
gnb = GaussianNB()
prediction = gnb.fit(dense,dense_classlabels).predict(test)
print(len(prediction))


test_labels = np.asarray(test_classlabels)
predictions = np.asarray(prediction)
# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
print(metrics.confusion_matrix(test_labels, predictions, labels=[1, 0]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(test_labels, predictions, labels=[1,0]))

