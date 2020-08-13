############LOGISTIC REGRESSION##################

# bussiness problem:whether a political candidate wins an election

#load dataset
election <- read.csv(file.choose())
View(election)

sum(is.na(election))
#5

election <- na.omit(election) # removing na values there is a total row consists of na values
election

colnames(election)

election <- election[-1]
attach(election)
#GLM function uses sigmoid curve to produce desirable outputs
#the output of sigmoid curve lies between 0 & 1


model <- glm('Result~Year+Amount.Spent+Popularity.Rank',data = election,family = "binomial")
summary(model)

# Null deviance: 1.3460e+01  on 9  degrees of freedom
#Residual deviance: 6.5897e-10  on 6  degrees of freedom
#AIC: 8
#we observe difference between null and residual dev to compare with different models

#to calculate the odds ratio we need to take the exp of coef(model)
exp(coef(model))

confusion <- table(prob>0.9,election$Result)
confusion
#      
#      0   1
#FALSE 4   0
#TRUE  0   6

accuracy <- sum(diag(confusion))/sum(confusion)
accuracy
#1

prob <- predict(model,election,type = 'response')
prob

#confusion matrix and considering threshold value 0.9
pred_val <- NULL
yes_no <- NULL

pred_val <- ifelse(prob>0.9,1,0)
yes_no <- ifelse(prob>0.9,'yes','no')

#create new column to store the above values
election[,'prob'] <- prob
election[,'pred_val'] <- pred_val
election[,'yes_no'] <- yes_no

View(election[,c(1,5:7)])

tab <- table(election$Result,election$pred_val)

#     0   1
#  0  4   0
#  1  0   6

#Classification and regression training(caret)
#caret contain the process to streamline the process for complex regression amd classification problem
library(caret)
confusionMatrix(tab)

#Confusion Matrix and Statistics

#     0   1
#  0  4   0
#  1  0   6


#Accuracy : 1          
#95% CI : (0.6915, 1)
#No Information Rate : 0.6        
#P-Value [Acc > NIR] : 0.006047   

#Kappa : 1          

#Mcnemar's Test P-Value : NA         
                                     
#            Sensitivity : 1.0        
#            Specificity : 1.0        
#         Pos Pred Value : 1.0        
#        Neg Pred Value : 1.0        
#             Prevalence : 0.4        
#         Detection Rate : 0.4        
#   Detection Prevalence : 0.4        
#      Balanced Accuracy : 1.0        
                                     
#       'Positive' Class : 0 

#roc curve is used to evakuate the betterness of the logistic model
#more the area under the curve better the model is
#we will use roc curve for any classifiaction technique not only for logistic model

library(ROCR)
rocrpred <- prediction(prob,election$Result)
rocrperf <- performance(rocrpred,'tpr','fpr')

str(rocrperf)

plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
#more the area under the curve better the model is
#getting cutoffs or threshold value along with true positive and false positive rate

str(rocrperf)
rocr_cutoff <- data.frame(cut_off=rocrperf@alpha.values[[1]],'fpr'=rocrperf@x.values,'tpr'=rocrperf@y.values)
colnames(rocr_cutoff) <- c('cut_off','fpr','tpr')
View(rocr_cutoff)

#dplyr is a powerfull package to transform and summarize tablular data with rows and columns
library(dplyr)
rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off,6)
#sorting dataframe with respect to tpr in decresing order
rocr_cutoff <- arrange(rocr_cutoff,desc(tpr))
View(rocr_cutoff)
