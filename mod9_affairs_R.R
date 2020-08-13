############LOGISTIC REGRESSION##################

#Applied Econometrics with R
install.packages('AER')
library('AER')
affairs <- read.csv(file.choose())
View(affairs)

sum(is.na(affairs))
#0

dim(affairs)
#601  19

colnames(affairs)

affairs <- affairs[,-1]#removing column 1 as it consists of index no's
colnames(affairs)

#converting the values >= 1 to 1 as it should be in the form of only yes or no 
affairs$naffairs[affairs$naffairs>0] <- 1
affairs$naffairs[affairs$naffairs==0] <- 0
View(affairs$naffairs)

attach(affairs)

#GLM function uses sigmoid curve to get desireable results
#the output of sigmoid curve lies in between 0 to 1
model <- glm(naffairs~.,data = affairs,family = 'binomial')
summary(model)

#remove variables with NA values
model <- glm(naffairs~kids+vryunhap+unhap+avgmarr+hapavg+antirel+notrel+slghtrel+smerel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4+yrsmarr5,data=affairs,family = 'binomial')
summary(model)
# Null deviance: 675.38  on 600  degrees of freedom
#Residual deviance: 602.21  on 586  degrees of freedom
#AIC: 632.21

#to calculate odds ratio we are going to take exp of coef(model)
exp(coef(model))
#remove smerel bcz it has highest p value

model1 <- glm(naffairs~kids+vryunhap+unhap+avgmarr+hapavg+antirel+notrel+slghtrel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4+yrsmarr5,data=affairs,family = 'binomial')
summary(model1)
# Null deviance: 675.38  on 600  degrees of freedom
#Residual deviance: 602.28  on 587  degrees of freedom
#AIC: 630.28

model2 <- glm(naffairs~kids+vryunhap+unhap+avgmarr+hapavg+antirel+notrel+slghtrel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4,data=affairs,family = 'binomial')
summary(model2)
#Null deviance: 675.38  on 600  degrees of freedom
#Residual deviance: 602.53  on 588  degrees of freedom
#AIC: 628.53

model3 <-  glm(naffairs~vryunhap+unhap+avgmarr+hapavg+antirel+notrel+slghtrel+yrsmarr1+yrsmarr2+yrsmarr3+yrsmarr4,data=affairs,family = 'binomial')
summary(model3)
# Null deviance: 675.38  on 600  degrees of freedom
#Residual deviance: 602.82  on 589  degrees of freedom
#AIC: 626.82

model4 <- glm(naffairs~vryunhap+unhap+avgmarr+hapavg+antirel+notrel+slghtrel+yrsmarr1+yrsmarr2+yrsmarr3,data=affairs,family = 'binomial')
summary(model4)
# Null deviance: 675.38  on 600  degrees of freedom
#Residual deviance: 603.33  on 590  degrees of freedom
#AIC: 625.33

model5 <-  glm(naffairs~vryunhap+unhap+avgmarr+hapavg+antirel+slghtrel+yrsmarr1+yrsmarr2,data=affairs,family = 'binomial')
summary(model5)
#Null deviance: 675.38  on 600  degrees of freedom
#Residual deviance: 607.49  on 592  degrees of freedom
#AIC: 625.49

#odds ratio
exp(coef(model5))


prob <- predict(model5,affairs,type='response')
prob

#confusion matrix and threshold value as 0.5
confusion <- table(prob>0.5,affairs$naffairs)
confusion

#model accuracy
accuracy <- sum(diag(confusion))/sum(confusion)
accuracy
#0.7770383

#creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
yes_no <- NULL

pred_values <- ifelse(prob>0.5,1,0)
yes_no <- ifelse(prob>0.5,'yes','no')

#creating new columns to store above values
affairs[,'prob'] <- prob
affairs[,'pred_values'] <- pred_values
affairs[,'yes_no'] <- yes_no

View(affairs[,c(1,19:21)])

table(affairs$naffairs,affairs$pred_values)


#calculate the below matrics
pred_values=as.factor(pred_values)
affairs$naffairs=as.factor(affairs$naffairs)
install.packages('e1071')
library('e1071')
install.packages('caret')
library('caret')

confusion=confusionMatrix(data=pred_values,reference = affairs$naffairs,positive = "1")
confusion

#Confusion Matrix and Statistics

#Reference
#Prediction   0   1
#         0 446 129
#         1   5  21

#Accuracy : 0.777           
#95% CI : (0.7416, 0.8097)
#No Information Rate : 0.7504          
#P-Value [Acc > NIR] : 0.07071         

#Kappa : 0.178           

#Mcnemar's Test P-Value : < 2e-16         
                                          
#           Sensitivity : 0.14000         
#           Specificity : 0.98891         
#        Pos Pred Value : 0.80769         
#        Neg Pred Value : 0.77565         
#            Prevalence : 0.24958         
#         Detection Rate : 0.03494         
#   Detection Prevalence : 0.04326         
#      Balanced Accuracy : 0.56446         
                                  
#      'Positive' Class : 1  


library(ROCR)
rocrpred <- prediction(prob,affairs$naffairs)
rocrperf <- performance(rocrpred,'tpr','fpr')
str(rocrperf)
plot(rocrperf,colarize=T,text.adjacent=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained
## Getting cutt off or threshold value along with true positive and false positive rates in a data frame 
str(rocrperf)
rocr_cutoff <- data.frame(cut_off=rocrperf@alpha.values[[1]],'frp'=rocrperf@x.values,'tpr'=rocrperf@y.values)
colnames(rocr_cutoff) <- c('cutoff','FPR','TPR')
View(rocr_cutoff)

library(dplyr)
rocr_cutoff$cutoff <- round(rocr_cutoff$cutoff,6)
#sorting dataframe with respect to tpr indecresing order
rocr_cutoff <- arrange(rocr_cutoff,desc(TPR))
View(rocr_cutoff)
