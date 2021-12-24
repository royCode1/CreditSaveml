rm(list=ls())

#Some libraries
library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
library(ggplot2)
library(Hmisc)
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

Final<-read.csv(file.choose(), header=T)

Final <- Final %>% filter(educ != 4)
Final <- Final %>% filter(educ != 5)
Final <- Final %>% filter(account != 3)
Final <- Final %>% filter(q5 != 3)
Final <- Final %>% filter(q5 != 4)
Final <- Final %>% filter(q39 != 3)
Final <- Final %>% filter(q39 != 4)
Final <- Final %>% filter(q34 != 3)
Final <- Final %>% filter(q34 != 4)
Final <- Final %>% filter(q30 != 3)
Final <- Final %>% filter(q30 != 4)
Final <- Final %>% filter(q24 != 5)
Final <- Final %>% filter(q24 != 6)
Final <- Final %>% filter(q21a != 3)
Final <- Final %>% filter(q21a != 4)
Final <- Final %>% filter(q21c != 3)
Final <- Final %>% filter(q21c != 4)
Final <- Final %>% filter(q22a != 3)
Final <- Final %>% filter(q22a != 4)
Final <- Final %>% filter(q22b != 3)
Final <- Final %>% filter(q22b != 4)
Final <- Final %>% filter(q22c != 3)
Final <- Final %>% filter(q22c != 4)
Final <- Final %>% filter(q18a != 3)
Final <- Final %>% filter(q18a != 4)


###Picking up appropriate variables
FinData<- Final[,c(7:11,17,41,43:48,60,68,75,38)]
summary(FinData)
str(FinData)
colSums(is.na(FinData))

FinData <- na.omit(FinData)
summary(FinData)
#colSums(is.na(data1))
#str(data1)


###CREATING DUMMIES

#dummies for q18a variable
FinData$FIsaved<-ifelse(FinData$q18a==2,0,1)

#dummies for female variable
FinData$gender<-ifelse(FinData$female==2,0,1)

#dummies for account variable
FinData$yaccount<-ifelse(FinData$account==2,0,1)

#dummies for q5 variable
FinData$creditcard<-ifelse(FinData$q5==2,0,1)

#dummies for q21a variable
FinData$borrowfrombank<-ifelse(FinData$q21a==2,0,1)

#dummies for q21c variable
FinData$borrowfromfamily<-ifelse(FinData$q21c==2,0,1)

#dummies for q22a variable
FinData$ledu<-ifelse(FinData$q22a==2,0,1)

#dummies for q22b variable
FinData$lmedic<-ifelse(FinData$q22b==2,0,1)

#dummies for q22c variable
FinData$lfarm<-ifelse(FinData$q22c==2,0,1)

#dummies for q24 variable
FinData$emergency<-ifelse(FinData$q24==1 | FinData$q24==2,0,1)

#dummies for q30 variable
FinData$utility<-ifelse(FinData$q30==2,0,1)

#dummies for q34 variable
FinData$wage<-ifelse(FinData$q34==2,0,1)

#dummies for q39 variable
FinData$gtrans<-ifelse(FinData$q39==2,0,1)

FinData$q24<-NULL
View(FinData)
names(FinData)
FinData1 <- FinData[,c(2:4,17:29)]
#View(data2)


#Partition dataset into train and val
set.seed(1719)
pd<-sample(2,nrow(FinData1),replace=TRUE, prob=c(0.7,0.3))

train<-FinData1[pd==1,]
val<-FinData1[pd==2,]


#Normalize age in the train dataset
normalize<-function(x){
  +return((x-min(x))/(max(x)-min(x)))}

train<-transform(train, age=ave(train$age ,FUN = normalize))
View(train)


val<-transform(val, age=ave(val$age ,FUN = normalize))
View(val)

#Logistic Regression on all variables
#install.packages(c("SDMTools","pROC", "Hmisc"))
library(SDMTools)
library(pROC)
library(Hmisc)

train.logit<-train
val.logit<-val
names(train)

###Try1
Logit.eq.1<-FIsaved  ~ age+educ+inc_q+emergency+gender+yaccount+creditcard+borrowfrombank+borrowfromfamily+ledu+lmedic+lfarm+utility+wage+gtrans
Logit.1 <- glm(Logit.eq.1  , train.logit, family = binomial)
summary(Logit.1)
vif(Logit.1)

###Try2
Logit.eq.2<-FIsaved  ~ age+educ+inc_q+emergency+yaccount+creditcard+borrowfrombank+borrowfromfamily+ledu+lmedic+lfarm+utility+wage+gtrans
Logit.final <- glm(Logit.eq.2  , train.logit, family = binomial)
summary(Logit.final)
vif(Logit.final)

###Try3
Logit.eq.3<-FIsaved  ~ educ+inc_q+emergency+yaccount+creditcard+borrowfrombank+borrowfromfamily+ledu+lfarm+utility+wage+gtrans
Logit.final <- glm(Logit.eq.3  , train.logit, family = binomial)
summary(Logit.final)
vif(Logit.final)


###Provides val dataset into the model and gives Predictions
varImp(Logit.final)
pred.logit.final <- predict.glm(Logit.final, newdata=val.logit, type="response")
pred.logit.final


###Gives the confusion matrix and accuracy of the model
tab.logit<-table(val.logit$FIsaved,pred.logit.final>0.5)
tab.logit
accuracy.logit<-sum(diag(tab.logit))/sum(tab.logit)
accuracy.logit


#Mean Square Error
m=(pred.logit.final-val.logit$FIsaved)^2
summary(m)

##Histogram
hist(pred.logit.final)



#Reciever Operating Characteristic (ROC) and Area Under the Curve (AUC)
#detach(package:neuralnet)
library(pROC)
pred= prediction(pred.logit.final, val.logit$FIsaved)
roc=performance(pred,"tpr","fpr")
plot(roc, colorize=T, main="ROC curve")
auc=performance(pred,'auc')
auc=unlist(slot(auc,"y.values"))
auc=round(auc,4)
legend(.6,.4,auc, title="AUC")
abline(h=0.5)

####K-Fold
set.seed(123)
folds<-createFolds(FinData1$FIsaved,k=3)
str(folds)


Eq.2<-FIsaved  ~ age+educ+inc_q+emergency+gender+yaccount+creditcard+borrowfrombank+borrowfromfamily+ledu+lmedic+lfarm+utility+wage+gtrans
cv_logit<-lapply(folds,function(x){ 
  Train<-train[x,]
  Test<-val[-x,]
  Logit.1<-glm(Eq.2, Train, family = binomial)
  pred.1 <- predict.glm(Logit.1, newdata=Test, type="response")
  actual<-Test$FIsaved
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
pred= prediction(pred.logit.final, val.logit$FIsaved)
eval= performance(pred, "acc")
plot(eval)
ind = which.max( slot(eval, "y.values")[[1]] )
acc = slot(eval, "y.values")[[1]][ind]
cutoff = slot(eval, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))

##SUbset
library(ISLR)

library(leaps)
Linear<-FIsaved  ~ age+educ+inc_q+emergency+gender+yaccount+creditcard+borrowfrombank+borrowfromfamily+ledu+lmedic+lfarm+utility+wage+gtrans
regfitfull<-regsubsets(Linear, data=FinData1)
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

x.G=model.matrix(FIsaved~., data=FinData1)[,-4]
y.G=na.omit(FinData1$FIsaved)

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
#plot(cv.out.ridge)
#names(cv.out.ridge)
bestlambda.ridge.G=cv.out.ridge.G$lambda.min
bestlambda.ridge.G
pred.ridge.G<-predict(ridge.model.G,  s=bestlambda.ridge.G, newx=x.G[test.Ridge.G,])
MSERidge.G<-mean((pred.ridge.G-y.test.GR)^2)
MSERidge.G
pred.ridge.G<-predict(ridge.model.G, type="coefficients", s=bestlambda.ridge.G, newx=x.G[test.Ridge.G,])
pred.ridge.G
plot(ridge.model.G)
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
plot(lasso.model.G)
plot(cv.out.lasso.G)
