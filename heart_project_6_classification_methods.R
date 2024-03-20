install.packages('neuralnet',dependencies = T)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(pROC)
library(caret)
library(MASS)
library(randomForest)
library(neuralnet)

# read the dataset


heart.df <- read.csv("heart.csv")
head(heart.df)


## find if there any missing values in the datset

sum(!complete.cases(heart.df))

missing_counts <- colSums(is.na(heart.df));missing_counts

total_missing <- sum(is.na(heart.df));total_missing

## there are no missing values in the data set taken

## run the summary to understand

summary(heart.df)

## the categorical variables are - sex, fbs, output, 

heart.numeric <- heart.df[,c(1,2,3,4,5,6,7,8,9,10,11,12,13)]
for (i in 1:14) {
  boxplot(heart.df[,i], main=names(heart.df[i]), xlab="", col="skyblue")
}


##Plot the histogram for age 

## before doing that we need to convert the "sex" variable which is perceived as numerical into a  categorical one.

heart.df$sex <- factor(heart.df$sex)
## heart.df$sex<- as.numeric(heart.df$sex)


ggplot(data = heart.df, aes(x = age, fill = sex)) + 
  geom_histogram(binwidth=1,colour = 'white') 

## histogram for resting heartbeat with the divisions on sex

ggplot(data = heart.df, aes(x = trtbps, fill = sex)) + 
  geom_histogram(binwidth=2,colour = 'white') 

ggplot(data = heart.df, aes(x = trtbps, fill = sex)) + 
  geom_histogram(binwidth = 2, color = 'white') +
  labs(title = "Histogram of resting heartbeat by Sex", x = "heart beat", y = "Count") +
  theme_minimal()


##scatter plot for the age and cholesterol levels

qplot(heart.df$age, heart.df$chol, data = heart.df, colour = heart.df$sex)


##Data Modelling

## Splitting the dataset:

set.seed(123)  
train.index <- sample(c(1:dim(heart.df)[1]), dim(heart.df)[1]*0.8)  
train.df <- heart.df[train.index, ]
valid.df <- heart.df[-train.index, ]

## Run logistic Regression on the training dataset 


heart.train.logit.full <- glm(output~., data=train.df, family="binomial")
summary(heart.train.logit.full)

formula(heart.train.logit.full)


empty.heart.train<- glm(output~1,data=heart.df,family="binomial")
summary(empty.heart.train)


heart.train.logit.full$coefficients   ## coefficients of the full logistic regression

## Trying to find the best model with lowest AIC values

#######  Trying forward model construction

forwards = stepAIC(empty.heart.train,scope=list(lower=formula(empty.heart.train),upper=formula(heart.train.logit.full)), direction="forward",trace=0)

formula(forwards)
summary(forwards)

stepwise = stepAIC(empty.heart.train,scope=list(lower=formula(empty.heart.train),upper=formula(heart.train.logit.full)), direction="both",trace=1)
formula(stepwise)
summary(stepwise)



#######  Trying backward model construction
backwards.AIC = stepAIC(heart.train.logit.full)
summary(backwards.AIC)
formula(backwards.AIC)



#####  Confusion Matrix --  using the validation dataset to find the accuracy rate

heart.train.logit.full.predict <- predict(heart.train.logit.full,valid.df,type="response")
heart.train.logit.full.classes <- ifelse(heart.train.logit.full.predict > 0.5, 1, 0)
confusionMatrix(as.factor(heart.train.logit.full.classes), as.factor(valid.df$output))

####  -  Step wise FORWARD model
 
    #####  Confusion matrix for forward formula

forwards.predict <- predict(forwards,valid.df,type="response")
forwards.predict.classes <- ifelse(forwards.predict>0.5,1,0)
confusionMatrix(as.factor(forwards.predict.classes), as.factor(valid.df$output))

    #####  Confusion matrix for BOTH step formula

stepwise.predict <- predict(stepwise,valid.df,type="response")
stepwise.predict.classes <- ifelse(stepwise.predict>0.5,1,0)
confusionMatrix(as.factor(stepwise.predict.classes), as.factor(valid.df$output))


## stepAIC backwards

backwards.AIC.predict <- predict(backwards.AIC, valid.df, type = "response")
backwards.AIC.predict.classes <- ifelse(backwards.AIC.predict > 0.5, 1, 0)
confusionMatrix(as.factor(backwards.AIC.predict.classes), as.factor(valid.df$output))





####### Decision Tree

heart.tree <- rpart(output~., data=train.df, method="class")
prp(heart.tree, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = 10)
prp(heart.tree, type = 2, extra = "auto", under = TRUE, split.font = 1, varlen = -10)
printcp(heart.tree)

## construct the confusion matrix of the tree.

heart.tree.valid.predict <- predict(heart.tree,valid.df,type = "class")
confusionMatrix(heart.tree.valid.predict, as.factor(valid.df$output))

## construct a deeper tree

deeper.tree <- rpart(output~.,data=train.df,method="class", cp = -1, minsplit = 2)

prp(deeper.tree, type = 2, extra = "auto", under = TRUE, split.font = 1, varlen = -10)  ## the deeper tree plot

deeper.tree.valid.predict <- predict(deeper.tree,valid.df,type = "class")  ## checking the classification with the valid dataset
confusionMatrix(deeper.tree.valid.predict, as.factor(valid.df$output))


## the deeper tree that model that has all the variables included has lesser accuracy than the first constructed tree. 
##       Hence the tree needs pruning

## we shall prune the tree using Cp factor

cv.ct <- rpart(output ~ ., data = train.df, method = "class", cp = 0.000001, minsplit = -1, xval = 5)
printcp(cv.ct)
pruned.ct <- prune(cv.ct, cp =   0.0136364 )  ## the best accuracy is achieved at this cp value.

printcp(pruned.ct)
prp(pruned.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white')) 

pruned.tree <- predict(pruned.ct,valid.df,type = "class")
confusionMatrix(pruned.tree, as.factor(valid.df$output))

cv.ct.predict <- predict(cv.ct,valid.df, type="class")
confusionMatrix(cv.ct.predict, as.factor(valid.df$output))

###  Constructing the Random Forest

heart.randomforest <- randomForest(as.factor(output)~.,data=train.df,ntree=1000,mtry=4,nodesize=4,importance=TRUE,bootstrap=TRUE)

varImpPlot(heart.randomforest,type=1)

## confusion matrix for the Random Forest

heart.randomforest.predict <- predict(heart.randomforest,valid.df)
confusionMatrix(heart.randomforest.predict,as.factor(valid.df$output))



######### Construct a neural network model

nn.heart <- neuralnet(output ~., data = train.df, linear.output = F, hidden = 5,learningrate=1.5)

plot(nn.heart, rep="best")

nn.pred <- predict(nn.heart, valid.df, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred.classes), as.factor(valid.df$output))






############ Model Selection



#####  Area under the curve 

r.logit.full.logit <- roc(as.numeric(valid.df$output),as.numeric(heart.train.logit.full.predict))

plot.roc(r.logit.full.logit, xlab="Specificity for logit model", ylab="Sensitivity for logit", main="ROC curve for logistic regression")

auc(r.logit.full.logit)


######  ROC for the Forward regression

r.logit.forward <- roc(as.numeric(valid.df$output),as.numeric(forwards.predict.classes))

plot.roc(r.logit.forward, xlab="Specificity for forward logit model", ylab="Sensitivity for forward logit", main="ROC curve for forward logistic regression")

auc(r.logit.forward)

#### ROC for the stepwise regression
r.logit.stepwise <- roc(as.numeric(valid.df$output),as.numeric(stepwise.predict.classes))

plot.roc(r.logit.stepwise, xlab="Specificity for stepwise logit model", ylab="Sensitivity for stepwise logit", main="ROC curve for stepwise logistic regression")

auc(r.logit.stepwise)

### ROC FOR BACKWARD  Step AIC function

r.logit.backwards <- roc(as.numeric(valid.df$output),as.numeric(backwards.predict.classes))

plot.roc(r.logit.backwards, xlab="Specificity for backwards logit model", ylab="Sensitivity for backwards logit", main="ROC curve for backwards logistic regression")

auc(r.logit.backwards)




### ROC for the  pruned tree

r.tree.not.pruned <-roc(as.numeric(valid.df$output),as.numeric(heart.tree.valid.predict))

plot.roc(r.tree.not.pruned, xlab="Specificity for decision tree model", ylab="Sensitivity for decision tree model", main="ROC curve for decision tree model")

auc(r.tree.not.pruned)

#### ROC for the pruned tree

r.tree.pruned <-roc(as.numeric(valid.df$output),as.numeric(heart.tree.valid.predict))

plot.roc(r.tree.pruned,xlab="Specificity for pruned tree model", ylab="Sensitivity for pruned tree model", main="ROC curve for pruned tree model")


auc(r.tree.pruned)

##### ROC for the random Forest

r.randomforest <- roc(as.numeric(valid.df$output),as.numeric(heart.randomforest.predict))

plot.roc(r.randomforest,xlab="Specificity for randomforest model", ylab="Sensitivity for randomforest model", main="ROC curve for randomforest model")

auc(r.randomforest)

I######### ROC for neural network model

r.nn <- roc(as.numeric(valid.df$output),as.numeric(nn.pred.classes))

plot.roc(r.nn,xlab="Specificity for neural network model", ylab="Sensitivity for neural network model", main="ROC curve for neural network model")

auc(r.nn)


formula(forwards)
formula(stepwise)
formula(heart.train.logit.full)
