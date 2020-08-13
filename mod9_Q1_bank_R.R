###################Logistic regression########################

#Load dataset
bank_data <- read.csv(file.choose())
sum(is.na(bank_data))

#preparing a linear regression model
mod_lm <- lm(y~.,data=bank_data)# '.' represent all the dependent variables
pred1 <- predict(mod_lm,bank_data)
pred1
plot(pred1)

#GLM functions uses sigmoid curve to produce desirable results
#the output of sigmoid function lies in between 0 and 1
model <- glm(y~.,data = bank_data,family = 'binomial')
summary(model)

# Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22640  on 45183  degrees of freedom
#AIC: 22696


#to calculate odds ratio we are going to calculate exp of coeff(model)
exp(coef(model))

model1 <- glm(y~age+default+balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+joentrepreneur+johousemaid+jomanagement+joretired+joself.employed+joservices+jostudent+jotechnician+jounemployed,data = bank_data,family = 'binomial')
summary(model1)
# Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22640  on 45183  degrees of freedom
#AIC: 22696

model2 <- glm(y~age+default+balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+joentrepreneur+johousemaid+jomanagement+joretired+joservices+jostudent+jotechnician+jounemployed,data = bank_data,family = 'binomial')
summary(model2)
# Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22640  on 45184  degrees of freedom
#AIC: 22694

model3 <- glm(y~default+balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+joentrepreneur+johousemaid+jomanagement+joretired+joservices+jostudent+jotechnician+jounemployed,data = bank_data,family = 'binomial')
summary(model3)
#Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22640  on 45185  degrees of freedom
#AIC: 22692

model4 <- glm(y~default+balance+housing+loan+duration+campaign+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+joentrepreneur+johousemaid+jomanagement+joretired+joservices+jostudent+jotechnician+jounemployed,data = bank_data,family = 'binomial')
summary(model4)
#Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22640  on 45186  degrees of freedom
#AIC: 22690

model5<- glm(y~default+balance+housing+loan+duration+campaign+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+joentrepreneur+johousemaid+jomanagement+joretired+jostudent+jotechnician+jounemployed,data = bank_data,family = 'binomial')
summary(model5)
# Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22640  on 45187  degrees of freedom
#AIC: 22688

model6<- glm(y~default+balance+housing+loan+duration+campaign+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+johousemaid+jomanagement+joretired+jostudent+jotechnician+jounemployed,data = bank_data,family = 'binomial')
summary(model6)
#Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22641  on 45188  degrees of freedom
#AIC: 22687

model7 <- glm(y~default+balance+housing+loan+duration+campaign+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+johousemaid+jomanagement+joretired+jostudent+jotechnician,data = bank_data,family = 'binomial')
summary(model7)
#Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22642  on 45189  degrees of freedom
#AIC: 22686
model8 <- glm(y~balance+housing+loan+duration+campaign+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+johousemaid+jomanagement+joretired+jostudent+jotechnician,data = bank_data,family = 'binomial')
summary(model8)
#Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22643  on 45190  degrees of freedom
#AIC: 22685

model9 <- glm(y~balance+housing+loan+duration+campaign+previous+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+johousemaid+jomanagement+joretired+jostudent,data = bank_data,family = 'binomial')
summary(model9)
# Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22645  on 45191  degrees of freedom
#AIC: 22685
model10 <-  glm(y~balance+housing+loan+duration+campaign+poutfailure+poutother+poutsuccess+con_cellular+con_telephone+divorced+married+joadmin.+joblue.collar+johousemaid+jomanagement+joretired+jostudent,data = bank_data,family = 'binomial')
summary(model10)

# Null deviance: 32631  on 45210  degrees of freedom
#Residual deviance: 22648  on 45192  degrees of freedom
#AIC: 22686

##we use null and residual deviance to compare between the models
#residual dev < null deviance so the model built is good
##as the model residual value is lesser than the null deviance we can say that our new model is good

#change in the AIC is greaterthan 2 
#there is a slight change in the AIC value for the model we build and the first model 

 

#to calculate odds ratio we are going to calculate exp of coeff(model)
exp(coef(model10)) 


prob <- predict(model10,bank_data,type = 'response')
prob

#confusion matrix and considering threshold value as 0.5
confusion <- table(prob>0.5,bank_data$y)
confusion

#model accuracy
accuracy <- sum(diag(confusion/sum(confusion)))
accuracy
# 0.9007764

#creating empty vectors to store  predicted class based on treshold value
pred_val <- NULL
yes_no <- NULL

pred_val <- ifelse(prob>0.5,1,0)
yes_no <- ifelse(prob>0.5,'yes','no')

#creating the new column to store the above values
bank_data[,"prob"] <- prob
bank_data[,"pred_val"] <- pred_val
bank_data[,'yes_no'] <- yes_no

View(bank_data[,c(32:35)])

table(bank_data$y,bank_data$pred_val)
#calculate the below matrix
#precision
tab <-table(bank_data$y,bank_data$pred_val)
tab

precision <- tab[1,1]/sum(tab[,1])
precision                          
# 0.915935

#sensitivity (recall) True positive rate
TPR<- tab[1,1]/sum(tab[1,])
TPR
# 0.9773308

#specificity or true negative rate
TNR <- tab[2,2]/sum(tab[2,])
TNR
# 0.3229344

FP_rate <- 1-TNR
FP_rate
#  0.6770656

FN_rate <- 1-TPR
FN_rate
# 0.0226692

F1 <- 2*((precision*TPR)/(precision+TPR))
F1
#0.9456374

# ROC Curve => used to evaluate the betterness of the logistic model
# more area under ROC curve better is the model 
# We will use ROC curve for any classification technique not only for logistic

library(ROCR)
rocrpred <-prediction(prob,bank_data$y)
rocrperf <- performance(rocrpred,'tpr','fpr')
str(rocrperf)
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained
# Getting cutt off or threshold value along with true positive and false positive rates in a data frame 
str(rocrperf)
rocr_cuttoff <- data.frame(cut_off=rocrperf@alpha.values[[1]],fpr=rocrperf@x.values,tpr=rocrperf@y.values)
colnames(rocr_cuttoff) <- c('cutoff','FPR','TPR')
View(rocr_cuttoff)

library(dplyr)
rocr_cuttoff$cutoff <- arrange(rocr_cuttoff,desc(TPR))
View(rocr_cuttoff)
