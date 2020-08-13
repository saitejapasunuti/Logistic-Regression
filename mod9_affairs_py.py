######################LOGISTIC REGRESSION###############

#####affaris dataframe################

#pandas is used for data manipulation ,cleaning ,analysis
import pandas as pd
#numpy deals with numerical data
import numpy as np

#load dataframe  
url_csv = 'D:\\360digiTMG\mod 9 logistic regression\mod 9 datasets\Affairs\Affairs.csv'
affairs1 = pd.read_csv(url_csv, index_col=0)
affairs1.head()#head method is used to display top 5 rows of the dataset

#convert into target column to 0 and 1
affairs1['affairs'] = (affairs1.naffairs > 0).astype(int) 
print(affairs1.isna().sum())

##model building 
import statsmodels.formula.api as sm
logit_model = sm.logit('affairs ~ kids+vryunhap+unhap+avgmarr+hapavg+antirel+notrel+slghtrel+smerel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4+yrsmarr5',data = affairs1).fit()

#summary
logit_model.summary()

#p vaules of most of the varibles are less than the significant values,except some ,so, we remoev them
logit_model1 = sm.logit('affairs ~ vryunhap+unhap+avgmarr+hapavg+antirel+slghtrel+yrsmarr1+yrsmarr2',data = affairs1).fit()
#summary
logit_model1.summary()

predict=logit_model1.predict(pd.DataFrame(affairs1))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
con_matrix=confusion_matrix(affairs1['affairs'],predict>0.5)
con_matrix

#array([[446,   5],
#       [129,  21]], dtype=int64)


print(classification_report(affairs1['affairs'],predict>0.5))

print(accuracy_score(affairs1['affairs'],predict>0.5))
#0.7770382695507487

print(precision_score(affairs1['affairs'],predict>0.5))
#0.8076923076923077

print(recall_score(affairs1['affairs'],predict>0.5))
#0.14

print(f1_score(affairs1['affairs'],predict>0.5))
#0.23863636363636365
#f1 score is close to 0 it creates imbalances in the model

#ROC curve 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc

auc=roc_auc_score(affairs1.affairs,predict)
auc
#area under the curve=0.7047376201034737

fpr,tpr,thresholds=roc_curve(affairs1.affairs,predict)

def plot_roc_curve(tpr,fpr):
    plt.plot(fpr,tpr,color='red',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',linestyle='--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic Curve')
    plt.legend()
    plt.show()
    
plot_roc_curve(tpr,fpr)    
