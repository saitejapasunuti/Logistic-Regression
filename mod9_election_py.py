##############LOGISTIC REGRESSION################

###ELECTION DATASET###

import pandas as pd
#pandas is used for data manipulation,cleansing and analysis

import numpy as np
#numpy deals with the numerical calculation

#load data
election=pd.read_csv('D:\\360digiTMG\mod 9 logistic regression\mod 9 datasets\election_data\election_data.csv')
election.head()

print(election.isna().sum())

election=election.drop('Election-id',axis=1)

election=election.drop([0],axis=0)
election    



election=election.rename(columns={'Amount Spent':'Amount_Spent','Popularity Rank':'Popularity_Rank'})
election


import statsmodels.formula.api as sm
import scipy as sp#scipy is used for scientific calculations

logit_model=sm.logit('Result~Year+Amount_Spent+Popularity_Rank',data=election).fit(method='bfgs')
#scipy's 'bfgs' is a good optimizer that works in many cases
logit_model.summary()

predict=logit_model.predict(pd.DataFrame(election))
predict

from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, classification_report

conf_matrix=confusion_matrix(election['Result'],predict>0.9)
conf_matrix

#array([[4, 0],
#       [0, 6]]

print(classification_report(election['Result'],predict>0.9))

print(accuracy_score(election['Result'],predict>0.9))
#1

print(f1_score(election['Result'],predict>0.9))
#1

print(precision_score(election['Result'],predict>0.9))
#1
print(recall_score(election['Result'],predict>0.9))
#1

#ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc

auc=roc_auc_score(election.Result,predict)
auc
#1
#area under curve is 1, more the area under the curve better the model is

fpr,tpr,threshold=roc_curve(election.Result,predict)

def plot_roc_curve(tpr,fpr):
    plt.plot(fpr,tpr,color='red',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',linestyle='--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic Curve')
    plt.legend()
    plt.show()
    
plot_roc_curve(tpr,fpr)
