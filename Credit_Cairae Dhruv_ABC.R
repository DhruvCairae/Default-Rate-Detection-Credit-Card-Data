credit <- read.csv("credit_default.csv")
dim(credit)
attach(credit)

library(tidyverse)
library(skimr)
library(glmnet)
library(ggplot2)
library(plotmo)
library(ROCR)
library(boot)
library(rpart)
library(rpart.plot)

skim(credit)
dim(credit)

colnames(credit)[colnames(credit) == 'default.payment.next.month'] <- 'default'
summary(credit)

credit$default<- as.factor(credit$default)
credit$EDUCATION<- as.factor(credit$EDUCATION)
credit$SEX<- as.factor(credit$SEX)
credit$MARRIAGE<- as.factor(credit$MARRIAGE)
credit$PAY_0<-as.factor(credit$PAY_0)

ggplot(credit, aes(x = default, fill = EDUCATION)) + 
  geom_bar()+
  theme_minimal()+
  labs(x = "Default:1, Education (1 = graduate school; 2 = university;
3 = high school; 4 = others)", y = "Count")

ggplot(credit, aes(x = default, fill = PAY_0)) + 
  geom_bar()+
  theme_minimal()+
  labs(x = "Default:1,-1&-2 = pay duly; 1 = payment delay for one month; 
       2 = payment delay for two months; . . .; 8 = payment delay for eight months", y = "Count")

credit$PAY_0<-as.integer(credit$PAY_0)
library(fastDummies)
credit.dummy <- dummy_cols(credit,
                           select_columns = "EDUCATION",
                           remove_first_dummy = TRUE,
                           remove_selected_columns = TRUE)
library(caTools)
set.seed(14004918)
split = sample.split(credit$default, SplitRatio = 0.8)
training_set = subset(credit, split == TRUE)
test_set = subset(credit, split == FALSE)

colnames(training_set)[colnames(training_set) == 'default.payment.next.month'] <- 'default'
summary(training_set)


ggplot(training_set, aes(x = default)) + 
  geom_bar(fill=c('green','red'))+
  theme_minimal()+
  labs(x = "Default:1", y = "Count")

hist(training_set$PAY_0,main="Repayment Status-First Month",freq = FALSE,col='blue')
lines(density(Boston$crim), lwd=5, col='blue')

full_model <- glm(default ~ ., family = binomial, data = training_set)
summary(full_model)
AIC(full_model)
BIC(full_model)

null_model <- glm(default ~ 1, family = binomial, data = training_set)
summary(null_model)
AIC(null_model)
BIC(null_model)

two_variable_model<- glm(default ~ EDUCATION+PAY_0, family = binomial, data = training_set)
summary(two_variable_model)
AIC(two_variable_model)
BIC(two_variable_model)

training.AIC.step <- step(full_model,data=training_set) #k=2, default AIC
summary(training.AIC.step)
AIC(training.AIC.step)
BIC(training.AIC.step)

training.BIC.step <- step(full_model,data=training_set,k=log(nrow(training_set))) #k=ln(n), BIC
summary(training.BIC.step)
AIC(training.BIC.step)
BIC(training.BIC.step)



index <- sample(nrow(credit),nrow(credit)*0.80)
credit_train = credit[index,]
credit_test = credit[-index,]
dummy <- model.matrix(~ ., data = credit)
credit_data_lasso <- data.frame(dummy[,-1])
credit_train_X = as.matrix(select(credit_data_lasso, -default)[index,])
credit_test_X = as.matrix(select(credit_data_lasso, -default)[-index,])
credit_train_Y = credit_data_lasso[index, "default"]
credit_test_Y = credit_data_lasso[-index, "default"]
credit_lasso <- glmnet(x=credit_train_X, y=credit_train_Y, family = "binomial")
credit_lasso_cv<- cv.glmnet(x=credit_train_X, y=credit_train_Y, family = "binomial", type.measure = "class")
plot(credit_lasso_cv)
coef(credit_lasso, s=credit_lasso_cv$lambda.min)
coef(credit_lasso, s=credit_lasso_cv$lambda.1se)

lasso_model_min<- glm(default ~LIMIT_BAL+
                        SEX+
                        EDUCATION+
                        MARRIAGE+
                        AGE+
                        PAY_0+
                        PAY_2+
                        PAY_3+
                        PAY_4+
                        PAY_5+
                        PAY_6+
                        BILL_AMT1+
                        BILL_AMT2+
                        BILL_AMT3+
                        BILL_AMT4+
                        BILL_AMT6+
                        PAY_AMT1+
                        PAY_AMT2+
                        PAY_AMT3+
                        PAY_AMT4+
                        PAY_AMT5+
                        PAY_AMT6
                        , family = binomial, data = training_set)
summary(lasso_model_min)
AIC(lasso_model_min)
BIC(lasso_model_min)

lasso_model_1se<- glm(default ~ LIMIT_BAL+
                        SEX+
                        EDUCATION+
                        MARRIAGE+
                        AGE+
                        PAY_0+
                        PAY_2+
                        PAY_3+
                        PAY_4+
                        PAY_5+
                        PAY_6+
                        BILL_AMT1+
                        PAY_AMT1+
                        PAY_AMT2+
                        PAY_AMT3+
                        PAY_AMT4+
                        PAY_AMT5+
                        PAY_AMT6
                        
                      , family = binomial, data = training_set)
summary(lasso_model_1se)
AIC(lasso_model_1se)
BIC(lasso_model_1se)

plot(lasso_fit, xvar = "lambda", label = TRUE)
plot_glmnet(lasso_fit, label=TRUE)  
plot_glmnet(lasso_fit, label=8, xvar ="norm")   
plot(cv_lasso_fit)

# Best Model - training.AIC.step

pred_resp <- predict(training.AIC.step,type="response")
table(training_set$default, (pred_resp > 0.5)*1, dnn=c("Truth","Predicted"))
table(training_set$default, (pred_resp > 1/6)*5, dnn=c("Truth","Predicted"))



pred_glm0_train<- predict(training.AIC.step, type="response")
pred <- prediction(pred_glm0_train, training_set$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE,main="In-Sample ROC Curve")
unlist(slot(performance(pred, "auc"), "y.values"))

cost2 <- function(r, pi)
   { weight1 = 5
weight0 = 1
pcut <- 1/(weight1/weight0+1) # Bayes estimator
c1 = (r == 1) & (pi < pcut)
#logical vector - true if actual 1 but predict 0
c0 = (r == 0) & (pi > pcut)
#logical vector - true if actual 0 but predict 1
return(mean(weight1 * c1 + weight0 * c0))

} 

cost2(training_set$default, pred_resp)

pred_glm0_test<- predict(training.AIC.step, newdata = test_set, type="response")
table(test_set$default, (pred_glm0_test > 0.5)*1, dnn=c("Truth","Predicted"))
table(test_set$default, (pred_glm0_test > 1/6)*5, dnn=c("Truth","Predicted"))
cost2(test_set$default, pred_glm0_test)

pred <- prediction(pred_glm0_test, test_set$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE,main="Out of Sample ROC Curve")
unlist(slot(performance(pred, "auc"), "y.values"))

pred_glm0_test<- predict(training.AIC.step, newdata = credit, type="response")
table(credit$default, (pred_glm0_test > 0.5)*1, dnn=c("Truth","Predicted"))
table(credit$default, (pred_glm0_test > 1/6)*5, dnn=c("Truth","Predicted"))
cost2(credit$default, pred_glm0_test)

pred <- prediction(pred_glm0_test, credit$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE,main="Full Data ROC Curve")
unlist(slot(performance(pred, "auc"), "y.values"))

#Asymmetric cost, pcut=1/(5+1)
costfunc  <- function(obs, pred.p){
  weight1 <- 5   # define the weight for "true=1 but pred=0" (FN)
  weight0 <- 1    # define the weight for "true=0 but pred=1" (FP)
  pcut <- 1/(1+weight1/weight0)
  c1 <- (obs==1)&(pred.p < pcut)    # count for "true=1 but pred=0" (FN)
  c0 <- (obs==0)&(pred.p >= pcut)   # count for "true=0 but pred=1" (FP)
  cost <- mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end 

#AUC as cost
costfunc1 = function(obs, pred.p){
  pred <- prediction(pred.p, obs)
  perf <- performance(pred, "tpr", "fpr")
  cost =unlist(slot(performance(pred, "auc"), "y.values"))
  return(cost)
} 

credit_glm1<- glm(default~. , family=binomial, data=credit)
cv_result  <- cv.glm(data=credit, glmfit=credit_glm1, cost=costfunc, K=5) 
cv_result$delta[2]
cv_result1  <- cv.glm(data=credit, glmfit=credit_glm1, cost=costfunc1, K=5) 
cv_result1$delta[2]


credit_rpart <- rpart(formula = default ~ . , 
                      data = training_set, 
                      method = "class", 
                      parms = list(loss=matrix(c(0,5,1,0), nrow = 2)))
prp(credit_rpart, extra = 1,main="Classification Tree")

credit_train_prob_rpart = predict(credit_rpart, training_set, type="prob")

pred = prediction(credit_train_prob_rpart[,2], training_set$default)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE,main="In-Sample ROC Curve")

slot(performance(pred, "auc"), "y.values")[[1]]

cost <- function(r, phat){
  weight1 <- 5
  weight0 <- 1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}


credit_train.pred.tree1<- predict(credit_rpart, training_set, type="class")
table(training_set$default, credit_train.pred.tree1, dnn=c("Truth","Predicted"))
cost(training_set$default, predict(credit_rpart, training_set, type="prob"))


credit_test_prob_rpart = predict(credit_rpart, test_set, type="prob")

pred = prediction(credit_test_prob_rpart[,2], test_set$default)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE,main="Out of Sample ROC Curve")

slot(performance(pred, "auc"), "y.values")[[1]]

credit_test.pred.tree1<- predict(credit_rpart, test_set, type="class")
table(test_set$default, credit_test.pred.tree1, dnn=c("Truth","Predicted"))
cost(test_set$default, predict(credit_rpart, test_set, type="prob"))


credit_full_prob_rpart = predict(credit_rpart, credit, type="prob")

pred = prediction(credit_full_prob_rpart[,2], credit$default)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE,main="Full Model ROC Curve")

slot(performance(pred, "auc"), "y.values")[[1]]

cost(credit$default, predict(credit_rpart, credit, type="prob"))




