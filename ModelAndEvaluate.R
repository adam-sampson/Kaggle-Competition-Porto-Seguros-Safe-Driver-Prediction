##########
# Kaggle Competition
# Porto Seguro's Safe Driver Prediction
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43282
# Author: Adam Sampson
##########
# This file is intended to do raw exploration of the data.
# It has been separated to clear up the actual Model code.
##########

source("LoadPackages.R")

packages <- c("data.table",
              "dplyr",
              #"dummies",
              "neuralnet",
              "stringr",
              #"ggplot2",
              "caret",
              "e1071",
              #"ROCR"
              "pROC"
              )
loadPackages(packages)
rm(packages)

source("Functions.R")

#---
# Import Data from files as data.table
#---
  ps.train.dt <- fread("train.csv")
  ps.test.dt <- fread("test.csv")
  gc(verbose = TRUE)
  
#---
# For testing purposes only, create smaller set to run faster. **REMOVE**
#--
  rm(ps.train2.dt)
  rm(ps.test2.dt)
  gc(verbose = TRUE)
  ps.train2.dt <- copy(rbind(sample_n(ps.train.dt[target==0],5000),
                       sample_n(ps.train.dt[target==1],500)))
  ps.test2.dt <- copy(sample_n(ps.test.dt,10000))
  rm(ps.train.dt)
  rm(ps.test.dt)
  gc()
  ps.train.dt <- copy(ps.train2.dt)
  ps.test.dt <- copy(ps.test2.dt)
  rm(ps.train2.dt)
  rm(ps.test2.dt)
  gc()

#---
# Convert columns to the correct classes
#---
  target.var <- c("target")
  categorical.var <- names(ps.train.dt) %>% str_subset(".+_cat")
  binary.var <- names(ps.train.dt) %>% str_subset(".+_bin")
  other.var <- ps.train.dt %>% select(-id,-target, -one_of(categorical.var), -one_of(binary.var)) %>% names()
  ordinal.var <- ps.train.dt %>% select(other.var) %>% select(1,2,3,4,5,6,8,13:26) %>% names()
  interval.var <- ps.train.dt %>% select(-id,-one_of(target.var),-one_of(categorical.var),
                                         -one_of(binary.var),-one_of(ordinal.var)) %>% names()
  rm(other.var)
  
  # Change target to factor
  changeCols <- target.var
  ps.train.dt[get(changeCols)==0,(changeCols) := "noClaim"]
  ps.train.dt[get(changeCols)==1,(changeCols) := "claim"]
  ps.train.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
  
#---
# Scale the variables
#---

#---
# Create a validation set
#---
  Train <- createDataPartition(ps.train.dt$target,p=0.8,list=FALSE)
  train.dt <- ps.train.dt[Train,]    
  validate.dt <- ps.train.dt[-Train,]
  rm(Train)
    
#---
# Build a model with all of the variables to get a baseline
#---
  # Set up resampling since we are imbalanced
  trainCtrl <- trainControl(method="cv",
                            #summaryFunction = twoClassSummary, 
                            #classProbs = TRUE,
                            savePredictions = TRUE)
  
  # general logistic regression
  allVarLogReg.mod <- train(target~., data=train.dt[,-1], 
                            method="glm", family="binomial",
                            trControl=trainCtrl)
    # attributes(allVarLogReg.mod)
    allVarLogReg.mod$finalModel
    allVarLogReg.mod$results
    varImp(allVarLogReg.mod)
    
    allVarLogReg.mod$pred
    
  # now use model to predict on validation set
  predictLogReg <- predict(allVarLogReg.mod, newdata = validate.dt)
    confusionMatrix(predictLogReg,validate.dt$target)  
  predictLogReg <- predict(allVarLogReg.mod, newdata = validate.dt,type = 'prob')
    logReg.ROC <- roc(predictor=predictLogReg$`1`,
                      response = validate.dt$target)
    plot(logReg.ROC,main="LogReg ROC")
    logReg.ROC$auc
    