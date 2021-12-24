rm(list=ls())
library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
library(ggord)
library(ggplot2)
library(Hmisc)
library(klaR)``
library(klaR)
library(MASS)
library(nnet)
library(plyr)
library(pROC)
library(psych)
library(scatterplot3d)
library(SDMTools)
library(dplyr)
library(ElemStatLearn)
library(rpart)
library(rpart.plot)
library(randomForest)
#library(neuralnet)

Find<-read.csv(file.choose(), header=T)
str(Find)
summary(Find)

Fx=Find[,c(7:11,14,17,34:37,40,45:48,60,68,75,85,86)]
str(Fx)

#Filtering Dataset
library(dplyr)
Fx <- Fx %>% filter(educ != 4)
Fx <- Fx %>% filter(educ != 5)
Fx <- Fx %>% filter(q2 != 3)
Fx <- Fx %>% filter(q2 != 4)
Fx <- Fx %>% filter(q5 != 3)
Fx <- Fx %>% filter(q5 != 4)
Fx <- Fx %>% filter(q39 != 3)
Fx <- Fx %>% filter(q39 != 4)
Fx <- Fx %>% filter(q34 != 3)
Fx <- Fx %>% filter(q34 != 4)
Fx <- Fx %>% filter(q30 != 3)
Fx <- Fx %>% filter(q30 != 4)
Fx <- Fx %>% filter(q24 != 5)
Fx <- Fx %>% filter(q24 != 6)
Fx <- Fx %>% filter(q22a != 3)
Fx <- Fx %>% filter(q22a != 4)
Fx <- Fx %>% filter(q22b != 3)
Fx <- Fx %>% filter(q22b != 4)
Fx <- Fx %>% filter(q22c != 3)
Fx <- Fx %>% filter(q22c != 4)
Fx <- Fx %>% filter(q16 != 3)
Fx <- Fx %>% filter(q16 != 4)
Fx <- Fx %>% filter(q20 != 3)
Fx <- Fx %>% filter(q20 != 4)
Fx <- Fx %>% filter(q17a != 3)
Fx <- Fx %>% filter(q17a != 4)
Fx <- Fx %>% filter(q17b != 3)
Fx <- Fx %>% filter(q17b != 4)
Fx <- Fx %>% filter(q17c != 3)
Fx <- Fx %>% filter(q17c != 4)

#assigning Dummies to dataset
Fx$gender<-ifelse(Fx$female==1,1,0)
Fx$debitcard<-ifelse(Fx$q2==1,1,0)
Fx$creditcard<-ifelse(Fx$q5==1,1,0)
Fx$account<-ifelse(Fx$account==1,1,0)
Fx$gtrans<-ifelse(Fx$q39==1,1,0)
Fx$wage<-ifelse(Fx$q34==1,1,0)
Fx$utility<-ifelse(Fx$q30==1,1,0)
Fx$emergency<-ifelse(Fx$q24==1| Fx$q24==2,1,0)
Fx$ledu<-ifelse(Fx$q22a==1,1,0)
Fx$lmedic<-ifelse(Fx$q22b==1,1,0)
Fx$lfarm<-ifelse(Fx$q22c==1,1,0)
Fx$online<-ifelse(Fx$q16==1,1,0)
Fx$lhouse<-ifelse(Fx$q20==1,1,0)
Fx$sfarm<-ifelse(Fx$q17a==1,1,0)
Fx$sold<-ifelse(Fx$q17b==1,1,0)
Fx$sedu<-ifelse(Fx$q17c==1,1,0)
Fx$saved<-ifelse(Fx$saved==1,1,0)
Fx$borrowed<-ifelse(Fx$borrowed==1,1,0)

Fx[,c(1,6:19)]<-NULL
View(Fx)

Fx=na.omit(Fx)
colSums(is.na(Fx))

pd<-sample(2,nrow(Fx),replace=TRUE, prob=c(0.7,0.3))

train<-Fx[pd==1,]
val<-Fx[pd==2,]

#Normalize?
normalize<-function(x){
  +return((x-min(x))/(max(x)-min(x)))}



train<-transform(train, age=ave(train$age,FUN = normalize))

val<-transform(val, age=ave(val$age,FUN = normalize))
View(train)


#data frame
train.logit<-train
val.logit<-val

library(SDMTools)
library(pROC)
library(Hmisc)

##Try 1
Logit.eq.final<- creditcard  ~ gender + age + debitcard + inc_q + emergency + account + gtrans + wage + utility + ledu + lmedic + lfarm + online + lhouse + sfarm + sold + sedu + saved + borrowed

Logit.final <- glm (Logit.eq.final   , train.logit, family=binomial)

summary(Logit.final)

##Try 2 (Removing the non-significant variables)
Logit.eq.final<- creditcard  ~ gender + age + debitcard + inc_q + emergency + account + gtrans + wage + utility +  lmedic + lfarm + online + lhouse  + sold + sedu + saved + borrowed


Logit.final <- glm (Logit.eq.final   , train.logit, family=binomial)

summary(Logit.final)

##Try 3 (Removing the non-significant variables)
Logit.eq.final<- creditcard  ~ age  + debitcard + inc_q + emergency + account + gtrans + wage + utility +  lmedic + lfarm + online + lhouse  + sold  + saved + borrowed


Logit.final <- glm (Logit.eq.final   , train.logit, family=binomial)

summary(Logit.final)

vif(Logit.final)

varImp(Logit.final)

pred.logit.final <- predict.glm(Logit.final, newdata=val.logit, type="response")
pred.logit.final

#Classification
tab.logit<-table(val.logit$account,pred.logit.final>.5)
tab.logit
#Logit
accuracy.logit<-sum(diag(tab.logit))/sum(tab.logit)
accuracy.logit


#Mean Square Error
m=(pred.logit.final-val.logit$creditcard)^2
summary(m)

##Histogram
hist(pred.logit.final)


#Reciever Operating Characteristic (ROC) and Area Under the Curve (AUC)
detach(package:neuralnet)
library(pROC)
library(ROCR)
pred= prediction(pred.logit.final, val.logit$creditcard)
roc=performance(pred,"tpr","fpr")
plot(roc, colorize=T, main="ROC curve")
auc=performance(pred,'auc')
auc=unlist(slot(auc,"y.values"))
auc=round(auc,4)
legend(.6,.4,auc, title="AUC")
abline(h=0.5)



####K-Fold
set.seed(123)
folds<-createFolds(Fx$account,k=3)
str(folds)

library(tidyverse)

Eq.2<-creditcard  ~ gender + age + debitcard + inc_q + emergency + account + gtrans + wage + utility + ledu + lmedic + lfarm + online + lhouse + sfarm + sold + sedu + saved + borrowed
cv_logit<-lapply(folds,function(x){
  Train<-train[x,]
  Test<-val[-x,]
  Logit.1<-glm(Eq.2, Train, family = binomial)
  pred.1 <- predict.glm(Logit.1, newdata=Test, type="response")
  actual<-Test$creditcard
  p<-ifelse(pred.1>0.52,1,0)
  #p_class<-factor(p,levels = levels,(actual[["Class"]]))
  tab.logit.1<-table(actual,p)
  sum(diag(tab.logit.1))/sum(tab.logit.1)
})

str(cv_logit)
fit.logit<-mean(unlist(cv_logit))
fit.logit

# CUtoff and Accuracy Prediction
library(ROCR)
pred= prediction(pred.logit.final, val.logit$creditcard)
eval= performance(pred, "acc")
plot(eval)
ind = which.max( slot(eval, "y.values")[[1]] )
acc = slot(eval, "y.values")[[1]][ind]
cutoff = slot(eval, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))

##SUbset
library(ISLR)

library(leaps)
Linear<-creditcard  ~ gender + age + debitcard + inc_q + emergency + account + gtrans + wage + utility + ledu + lmedic + lfarm + online + lhouse + sfarm + sold + sedu + saved + borrowed

regfitfull<-regsubsets(Linear, data=Fx)
reg.summary<-summary(regfitfull)
reg.summary

names(reg.summary)
reg.summary$rsq
reg.summary$adjr2


Adrsq.opt<-which.max(reg.summary$adjr2)
Adrsq.opt
coef(regfitfull,Adrsq.opt)



##Ridge and Lasso
library(glmnet)

View(Fx)
x.G=model.matrix(creditcard~., data=Fx)[,-9]
y.G=na.omit(Fx$creditcard)


grid=10^(seq(10,-2,length=100))
grid


###Ridge: alpha=0
ridge.model.G=glmnet(x.G,y.G,alpha=0,lambda=grid)
set.seed(1)
train.Ridge.G<-sample(1:nrow(x.G),nrow(x.G)/2)
test.Ridge.G<-(-train.Ridge.G)
y.test.GR<-y.G[test.Ridge.G]
#cross val
set.seed(1)
cv.out.ridge.G=cv.glmnet(x.G[train.Ridge.G,],y.G[train.Ridge.G],alpha=0)
#plot(cv.out.ridge.G)
#names(cv.out.ridge)
bestlambda.ridge.G=cv.out.ridge.G$lambda.min
bestlambda.ridge.G
pred.ridge.G<-predict(ridge.model.G,  s=bestlambda.ridge.G, newx=x.G[test.Ridge.G,])
MSERidge.G<-mean((pred.ridge.G-y.test.GR)^2)
MSERidge.G
pred.ridge.G<-predict(ridge.model.G, type="coefficients", s=bestlambda.ridge.G, newx=x.G[test.Ridge.G,])
pred.ridge.G
plot(ridge.model.G,,label=T)
plot(cv.out.ridge.G)

###Lasso: alpha=1
lasso.model.G=glmnet(x.G,y.G,alpha=1,lambda=grid)
###Test Train
set.seed(1)
train.Lasso.G<-sample(1:nrow(x.G),nrow(x.G)/2)
test.Lasso.G<-(-train.Lasso.G)
y.test.GL<-y.G[test.Lasso.G]
#cross val
set.seed(1)
cv.out.lasso.G=cv.glmnet(x.G[train.Lasso.G,],y.G[train.Lasso.G],alpha=1)
#plot(cv.out.lasso)
#names(cv.out.lasso)
bestlambda.lasso.G=cv.out.lasso.G$lambda.min
bestlambda.lasso.G
pred.lasso.G<-predict(lasso.model.G,  s=bestlambda.lasso.G, newx=x.G[test.Lasso.G,])
MSELasso.G<-mean((pred.lasso.G-y.test.GL)^2)
MSELasso.G

lasso.coeff.G<-predict(lasso.model.G, type="coefficients", s=bestlambda.lasso.G, newx=x.G[test.Lasso.G,])
lasso.coeff.G
plot(lasso.model.G, label=T)
plot(cv.out.lasso.G)

