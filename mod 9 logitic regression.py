###############logistic regression##################

#pandas is used for data manipulation ,cleaning and analysis
import pandas as pd

#numpy is used for numerical calculations
import numpy as np

#load dataset
bank=pd.read_csv('D:\\360digiTMG\\mod 9 logistic regression\\mod 9 datasets\\bank_data\\bank_data.csv')
bank.head()

bank.describe()

bank=bank.rename(columns={"joadmin.":"joadmin","joblue.collar":"joblue_collar","joself.employed":"joself_employed"})

bank

##########splitting model into train and test data###########
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(bank,test_size=0.3)
#model building

import statsmodels.formula.api as sm
logit_model=sm.logit('y ~ age+default+balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin+joblue_collar+joentrepreneur+johousemaid+jomanagement+joretired+joself_employed+joservices+jostudent+jotechnician+jounemployed',data=bank).fit()

logit_model.summary()

#remove the columns with higher p value more than the significant value and build the model
logit_model1=sm.logit('y ~ default+balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin+joblue_collar+johousemaid+jomanagement+joretired+jostudent',data=bank).fit()
logit_model1.summary()

logit_model2=sm.logit('y ~ default+balance+housing+loan+duration+campaign+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin+joblue_collar+johousemaid+jomanagement+joretired+jostudent',data=bank).fit()
logit_model2.summary()

logit_model3=sm.logit('y ~ balance+housing+loan+duration+campaign+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin+joblue_collar+johousemaid+jomanagement+joretired+jostudent',data=bank).fit()
logit_model3.summary()

logit_model4=sm.logit('y ~ balance+housing+loan+duration+campaign+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin+joblue_collar+johousemaid+jomanagement+joretired+jostudent',data=bank).fit()
logit_model4.summary()

predict=logit_model4.predict(pd.DataFrame(test_data[['balance','housing','loan','duration','campaign','poutfailure','poutother','poutsuccess','con_cellular','con_telephone','divorced','married','joadmin','joblue_collar','johousemaid','jomanagement','joretired','jostudent']]))

from sklearn.metrics import confusion_matrix
con_matrix=confusion_matrix(test_data['y'],predict>0.5)

con_matrix



accuracy=(11626+537)/(11626+288+1113+537)
accuracy
#0.896711884399882


precision=(11626)/(11626+1113)
precision
#0.9126305047491954

recall=(11626)/(11626+288)
recall
#0.9758267584354541
specificity=(537)/(537+1113)
specificity
#0.32545454545454544

fp_rate=1-specificity
fp_rate
#0.6745454545454546

fn_rate=1-recall
fn_rate
#0.024173241564545922

F1=2*((precision*recall)/(precision+recall))
F1
#0.9431712164848092

#F1 value is close to 1 so it is a balanced model

