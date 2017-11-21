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
              "ROCR",
              "pROC",
              "C50",
              "naivebayes"
              )
loadPackages(packages)
rm(packages)

source("Functions.R")

set.seed(42)

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
  ps.train.dt[get(changeCols)==0,targetChar := "noClaim"]
  ps.train.dt[get(changeCols)==1,targetChar := "claim"]
  changeCols <- "targetChar"
  ps.train.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
    str(ps.train.dt$targetChar)
  ps.train.dt[,(changeCols) := lapply(.SD,relevel,"noClaim"), .SDcols = changeCols]
    str(ps.train.dt$targetChar)
  
  changeCols <- c(categorical.var)
  ps.train.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
  ps.test.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
  
  str(ps.train.dt)  
#---
# Scale the variables
#---

#---
# Create a validation set
#---
  Train <- createDataPartition(ps.train.dt$targetChar,p=0.8,list=FALSE)
  train.dt <- ps.train.dt[Train,]    
  validate.dt <- ps.train.dt[-Train,]
  rm(Train)
  str(train.dt)

#---
# Try modelling using all variables
#---
       
  #---
  # Build a logistic regression model with all of the variables to get a baseline
  #---
    set.seed(1000)
    # Set up resampling since we are imbalanced
    trainCtrl <- trainControl(method="cv", 
                              #summaryFunction = twoClassSummary, 
                              classProbs = TRUE,
                              savePredictions = TRUE,
                              sampling = "smote")
    
    # trainCtrl <- trainControl(method="repeatedcv", number=10, repeats=5 
    #                           #summaryFunction = twoClassSummary, 
    #                           #classProbs = TRUE,
    #                           savePredictions = TRUE)
    
    # general logistic regression using smote sampling (instead of upsampling)
    allVarLogReg.mod <- train(targetChar~., data=train.dt[,-c(1,2)], 
                              method="glm", family="binomial", 
                              metric = "ROC",
                              trControl=trainCtrl)
    saveRDS(allVarLogReg.mod,file="allVarLogReg.mod.RDS")
      # attributes(allVarLogReg.mod)
      allVarLogReg.mod$finalModel
      allVarLogReg.mod$results
      varImp(allVarLogReg.mod)
      plot(varImp(allVarLogReg.mod),main="LogReg - Variable Importance")
      
      # allVarLogReg.mod$pred
      
    # now use model to predict on validation set
    predictLogReg <- predict(allVarLogReg.mod, newdata = validate.dt)
      confusionMatrix(predictLogReg,validate.dt$targetChar)  
    predictLogReg <- predict(allVarLogReg.mod, newdata = validate.dt,type = 'prob')
      logReg.ROC <- roc(predictor=predictLogReg$claim,
                        response = validate.dt$targetChar)
      plot(logReg.ROC,main="LogReg ROC")
      logReg.ROC$auc
      # logReg.ROC$thresholds
      
    # Using ROCR package  
    #predictLogReg <- predict(allVarLogReg.mod, newdata = validate.dt)
      #confusionMatrix(predictLogReg,validate.dt$target)
      pred <- prediction(predictLogReg$noClaim,validate.dt$targetChar)
      perf <- performance(pred,measure = "tpr", x.measure = "fpr")
      plot(perf)
      abline(a=0,b=1)    
      
      cost.perf <- performance(pred,"cost")
      plot(cost.perf)
      
      acc.perf <- performance(pred, measure = "acc")
      plot(acc.perf)
      
      auc.perf <- performance(pred, measure = "auc")
      auc.perf@y.values
      
      pred.cut <- opt.cut(perf = perf, pred = pred)
      print(pred.cut)    
      cutoff <- pred.cut[3]
      # cutoff <- 0.08
      
      mergeValPred <- data.table(true = validate.dt$targetChar, probPred = predictLogReg$claim)
      mergeValPred[probPred >= cutoff,pred := "claim"]
      mergeValPred[probPred < cutoff,pred := "noClaim"]
      confusionMatrix(mergeValPred$true,mergeValPred$pred)
  
  #---
  # Build a C5.0 tree model with all of the variables to get a baseline
  #--- 
    set.seed(1000)
    # Set up resampling since we are imbalanced
    c50trainCtrl <- trainControl(method="cv", 
                              # summaryFunction = twoClassSummary, 
                               classProbs = TRUE,
                               savePredictions = TRUE,
                               sampling = "smote")
    
    grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )
    
    allVarC50.mod <- train(targetChar~., data=train.dt[,-c(1,2)], 
                              tuneGrid=grid,method="C5.0",
                              # metric = "ROC",
                              trControl=c50trainCtrl)
    saveRDS(allVarC50.mod,file="allVarC50.mod.RDS")
    allVarC50.mod$finalModel
    allVarC50.mod$results
    summary(allVarC50.mod)
    
    predictC50 <- predict(allVarC50.mod, newdata = validate.dt)
      confusionMatrix(predictC50,validate.dt$targetChar)
    predictC50 <- predict(allVarC50.mod, newdata = validate.dt, type = 'prob')
      C50.ROC <- roc(predictor=predictC50$claim,
                        response = validate.dt$targetChar)
      plot(C50.ROC,main="C50 tree ROC")
      C50.ROC$auc
      
  #---
  # Build a Naive Bayes model with all of the variables to get a baseline
  #---
    set.seed(1000)
    
    NBtrainCtrl <- trainControl(method="cv", 
                                 # summaryFunction = twoClassSummary, 
                                 classProbs = TRUE,
                                 savePredictions = TRUE,
                                 sampling = "smote")
  
    allVarNB <- train(targetChar~., data=train.dt[,-c(1,2)],
                            method = 'naive_bayes',
                            trControl=NBtrainCtrl)
    saveRDS(allVarNB,file="allVarNB.mod.RDS")
    summary(allVarNB)
    allVarNB$results
    varImp(allVarNB)
    plot(varImp(allVarNB))
    
    predictNB <- predict(allVarNB, newdata = validate.dt)
      confusionMatrix(predictNB,validate.dt$targetChar)
    predictNB <- predict(allVarNB, newdata = validate.dt, type='prob')
      NB.ROC <- roc(predictor=predictNB$claim,
                     response = validate.dt$targetChar)
      plot(NB.ROC,main='Naive Bayes ROC')
      NB.ROC$auc
              
  #---
  # Build a neural network using mxnet with all of the variables to get a baseline
  #---
      set.seed(1000)
      
      NNtrainCtrl <- trainControl(method="cv", 
                                  # summaryFunction = twoClassSummary, 
                                  classProbs = TRUE,
                                  savePredictions = TRUE,
                                  sampling = "smote")
      
      # NNgrid <- expand.grid(size=c(10),decay=c(0.1))
      NNgrid <- expand.grid(size=c(10))
      
      allVarNN <- train(targetChar~., data=train.dt[,-c(1,2)],
                        method = 'nnet',
                        # preProcess = c('scale'),
                        #tuneGrid = NNgrid,
                        trControl=NNtrainCtrl)
      saveRDS(allVarNN,file="allVarNN.mod.RDS")
      summary(allVarNN)
      allVarNN$results
      
      predictNN <- predict(allVarNN, newdata = validate.dt)
        confusionMatrix(predictNN,validate.dt$targetChar)
      predictNN <- predict(allVarNN, newdata = validate.dt, type='prob')
        NN.ROC <- roc(predictor=predictNN$claim,
                    response = validate.dt$targetChar)
      plot(NN.ROC,main='Neural Network ROC')
      NN.ROC$auc
#---
# Re-try modelling after Feature engineering
#---
  
  #---
  # Feature engineering (see Sampson_Explore_Data.R for exploration and rationalle)
  #---
    # str(ps.train.dt)
    
    # Start with a clean start...
    rm(list = ls())
    gc(verbose = TRUE)
    
    source("Functions.R")
    
    set.seed(42)
    
  #---
  # Import Data from files as data.table
  #---
    ps.train.dt <- fread("train.csv")
    ps.test.dt <- fread("test.csv")
    gc(verbose = TRUE)
    
  #---
  # For testing purposes only, create smaller set to run faster. **REMOVE**
  #--
    # rm(ps.train2.dt)
    # rm(ps.test2.dt)
    # gc(verbose = TRUE)
    # ps.train2.dt <- copy(rbind(sample_n(ps.train.dt[target==0],5000),
    #                            sample_n(ps.train.dt[target==1],500)))
    # ps.test2.dt <- copy(sample_n(ps.test.dt,10000))
    # rm(ps.train.dt)
    # rm(ps.test.dt)
    # gc()
    # ps.train.dt <- copy(ps.train2.dt)
    # ps.test.dt <- copy(ps.test2.dt)
    # rm(ps.train2.dt)
    # rm(ps.test2.dt)
    # gc()
  
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
    ps.train.dt[get(changeCols)==0,targetChar := "noClaim"]
    ps.train.dt[get(changeCols)==1,targetChar := "claim"]
    changeCols <- "targetChar"
    ps.train.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
    str(ps.train.dt$targetChar)
    ps.train.dt[,(changeCols) := lapply(.SD,relevel,"noClaim"), .SDcols = changeCols]
    str(ps.train.dt$targetChar)
    
    # changeCols <- c(categorical.var)
    # ps.train.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
    # ps.test.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
    
    # str(ps.train.dt)  
  
  # ####
  # # Feature Engineering!
  # #### 
  # # ps_car_13 correlates to ps_car_15(remove)
  # # ps_car_04_cat(remove) correlates to ps_car_13 and ps_car_12(remove)
  # # ps_reg_03(remove) correlates to ps_reg_02 and ps_reg_01(remove)
  # # ps_ind_14(remove) correlates to ps_ind_11_bin(remove) and ps_ind_12_bin
  # # ps_ind_16_bin(remove) negatively correlates to ps_ind_17_bin and ps_ind_18_bin(remove)
  # ####
  #   ps.train.dt <- ps.train.dt %>% select(-ps_car_15, 
  #                        -ps_car_04_cat,
  #                        -ps_car_12,
  #                        -ps_reg_03,
  #                        -ps_reg_01,
  #                        -ps_ind_14,
  #                        -ps_ind_11_bin,
  #                        -ps_ind_16_bin,
  #                        -ps_ind_18_bin)
  #   
  # #####
  # # ps_ind_06_bin appears to have 5-10% relation to 0 or 1 target out ##naive_bayes found important
  # # ps_ind_07_bin appears to have 5-10% relation to 0 or 1 target out ##naive_bayes found important
  # # ps_ind_08_bin appears to have 1-3% relation to 0 or 1 target out (remove)
  # # ps_ind_09_bin appears to have 1-2% relation to 0 or 1 target out (remove)
  # # ps_ind_10_bin appears to have 0% relation to 0 or 1 target out (remove)
  # # ps_ind_11_bin appears to have 0% relation to 0 or 1 target out (remove)
  # # ps_ind_12_bin appears to have <1% relation to 0 or 1 target out (remove)
  # # ps_ind_13_bin appears to have 0% relation to 0 or 1 target out (remove)
  # # ps_ind_16_bin appears to have 3-6% relation to 0 or 1 target out
  # # ps_ind_17_bin appears to have 2-5% relation to 0 or 1 target out
  # # ps_ind_18_bin appears to have 1% relation to 0 or 1 target out (remove)
  # # ps_calc_15_bin appears to have 0% relation to 0 or 1 target out (remove)
  # # ps_calc_16_bin appears to have 1% relation to 0 or 1 target out (remove)
  # # ps_calc_17_bin appears to have 0-1% relation to 0 or 1 target out ##naive_bayes found important (remove)
  # # ps_calc_18_bin appears to have 0-1% relation to 0 or 1 target out (remove)
  # # ps_calc_19_bin appears to have 1% relation to 0 or 1 target out (remove)
  # # ps_calc_20_bin appears to have 1% relation to 0 or 1 target out (remove)
  # ####
  #   ps.train.dt <- ps.train.dt %>% select(-ps_ind_08_bin,-ps_ind_09_bin,-ps_ind_10_bin,-ps_ind_12_bin,
  #                                         -ps_ind_13_bin,-ps_calc_15_bin,-ps_calc_16_bin,-ps_calc_17_bin,
  #                                         -ps_calc_18_bin,-ps_calc_19_bin,-ps_calc_20_bin)
  #   
  #   
  # ####
  # # ps_ind_02_cat values 1 and -1/2 appear to have 2-3% relation, 3/4 <1% relation. 
  # # ps_ind_04_cat values 1 and -1/2 appear to have 2-3% relation
  # # ps_ind_05_cat values 0 and -1/2/4/5 appear to have 3-6% relation, 1/3/5 very little
  # # ps_car_01_cat values 6/7 and -1/9/11 appear to have 3-6% relation, 0/1/5/8 less, and 1/2/3/10 very little ##naive_bayes found important
  #   
  # # ps_car_02_cat values 0 and 1 appear to have 2-4% relation, -1 very little (remove)
  # # ps_car_03_cat values -1 and 1 appear to have 3-6% relation, 0 <1% relation.
  # # ps_car_04_cat values 0 and 1/2/3/4/5/6/7/8/9 appear to have 3-6% relation. (removed earlier in correlation)
  # # ps_car_05_cat values -1 and 1/2 appear to have 3-6%
  # # ps_car_06_cat values 0/11 and 13/15/9/17, 1/4/14 and 2/5/7/8/10/11/12/16, 3/6 very little
  #   
  # # ps_car_07_cat values -1/0 and 1 appear to have 1-2% (remove)
  # # ps_car_08_cat values 0 and 1 appear to have 1-3% (remove)
  # # ps_car_09_cat values 0 and 1 appear to have 2-4%, -1/2/3/4 very little
  # # ps_car_10_cat appears to have very little impact. (remove)
  # # ps_car_11_cat needs a closer look (too many vars). Keep 41 and 104 as -1, keep 32,64,82,99,103 as -2, everything else 0.
  # ####
  #   ps.train.dt <- ps.train.dt %>% select(-ps_car_07_cat,-ps_car_08_cat,-ps_car_10_cat, -ps_car_02_cat)
  #   ps.train.dt[ps_ind_02_cat == -1, ps_ind_02_cat := 2]
  #   ps.train.dt[ps_ind_02_cat == 4, ps_ind_02_cat := 3]
  #   ps.train.dt[ps_ind_04_cat == -1, ps_ind_04_cat := 2]
  #   ps.train.dt[(ps_ind_05_cat == -1)|(ps_ind_05_cat == 2)|(ps_ind_05_cat == 4)|(ps_ind_05_cat == 5), ps_ind_05_cat := 2]
  #   ps.train.dt[(ps_ind_05_cat == 1)|(ps_ind_05_cat == 3)|(ps_ind_05_cat == 5), ps_ind_05_cat := 1]
  #   ps.train.dt[(ps_ind_05_cat != 2)&(ps_ind_05_cat != 1),ps_ind_05_cat:=0]
  #   ps.train.dt[(ps_car_01_cat==7),ps_car_01_cat:=6]
  #   ps.train.dt[(ps_car_01_cat==-1)|(ps_car_01_cat ==9)|(ps_car_01_cat==11),ps_car_01_cat:=-1]
  #   ps.train.dt[(ps_car_01_cat != -1)&(ps_car_01_cat != 6),ps_car_01_cat:=0]
  #   ps.train.dt[(ps_car_05_cat == 2),ps_car_05_cat := 1]
  #   ps.train.dt[(ps_car_06_cat == 11),ps_car_06_cat := 0]
  #   ps.train.dt[(ps_car_06_cat == 9)|(ps_car_06_cat == 13)|(ps_car_06_cat == 15)|(ps_car_06_cat == 17),ps_car_06_cat:=9]
  #   ps.train.dt[(ps_car_06_cat == 1)|(ps_car_06_cat == 4)|(ps_car_06_cat == 14),ps_car_06_cat:=1]
  #   ps.train.dt[(ps_car_06_cat != 1)&(ps_car_06_cat != 9), ps_car_06_cat:=2]
  #   ps.train.dt[(ps_car_09_cat != 0)&(ps_car_09_cat != 1),ps_car_09_cat := 2]
  #   ps.train.dt[(ps_car_11_cat == 41)|(ps_car_11_cat == 104),ps_car_11_cat:=-1]
  #   ps.train.dt[(ps_car_11_cat == 32)|(ps_car_11_cat == 64)|(ps_car_11_cat == 82)|(ps_car_11_cat == 99)|(ps_car_11_cat == 103),ps_car_11_cat := -2]
  #   ps.train.dt[(ps_car_11_cat != -1)&(ps_car_11_cat != -2),ps_car_11_cat := 0]
  #   
  #   
  # ####
  # # ordinal
  # # ind_01 keep some 0/1 and 3/4/5/6/7, 2
  # # ind_03 keep some 2/3/4 and 0/5/6/7/8, 1/9/10/11
  # # ind_14 remove
  # # ind_15 keep some 0/1/2/3/4/5/6/7 and 8/9/10/11/12/13
  # # reg_01 keep some 1/2/3/4/5 and 0/7/8/9, 6 ##naive_bayes found important (removed in correlation)
  # # reg_02 keep some 0/1/2/3 and 5/6/7/8/9/1/11/12/13/14/15/16/17/18, 4 ##naive_bayes found important
  # # car_11 remove
  # # calc_01 remove
  # # calc_02 unsure
  # # calc_03 unsure
  # # calc_04 remove
  # # calc_05 remove
  # # calc_06 unsure
  # # calc_07 unsure
  # # calc_08 remove
  # # calc_09 unsure
  # # calc_10 unsure
  # # calc_11 unsure
  # # calc_12 remove
  # # calc_13 remove
  # # calc_14 unsure
  # ####
  #   ps.train.dt <- ps.train.dt %>% select(-ps_car_11,-ps_calc_01,-ps_calc_02,-ps_calc_03,-ps_calc_04,
  #                                         -ps_calc_05,-ps_calc_06,-ps_calc_07,-ps_calc_08,-ps_calc_09,
  #                                         -ps_calc_10,-ps_calc_11,-ps_calc_12,-ps_calc_13,-ps_calc_14)
  #   ps.train.dt[ps_ind_01 == 1,ps_ind_01 := 0]
  #   ps.train.dt[(ps_ind_01 == 3)|(ps_ind_01 == 4)|(ps_ind_01 == 5)|(ps_ind_01 == 6)|(ps_ind_01 == 7)|(ps_ind_01 == 8),ps_ind_01 := 3]
  #   ps.train.dt[(ps_ind_03 == 3)|(ps_ind_03 == 4),ps_ind_03 := 2]
  #   ps.train.dt[(ps_ind_03 == 5)|(ps_ind_03 == 6)|(ps_ind_03 == 7)|(ps_ind_03 == 8),ps_ind_03 := 0]
  #   ps.train.dt[(ps_ind_03 != 2)&(ps_ind_03 != 0), ps_ind_03 := 1]
  #   ps.train.dt[(ps_ind_15 == 0)|(ps_ind_15 == 1)|(ps_ind_15 == 2)|(ps_ind_15 == 3)|(ps_ind_15 == 4)|(ps_ind_15 == 5)|(ps_ind_15 == 6)|(ps_ind_15 == 7),ps_ind_15 := 0]
  #   ps.train.dt[(ps_ind_15 != 0),ps_ind_15 := 8]
  #   ps.train.dt[(ps_reg_02 == 1)|(ps_reg_02 == 2)|(ps_reg_02 == 3),ps_reg_02 := 0]
  #   ps.train.dt[(ps_reg_02 != 0)&(ps_reg_02 != 4), ps_reg_02 := 5]
  #       
  # ####
  # # ps_reg_03 visibal difference where 0 doesn't overlap ##naive_bayes found important
  # # ps_car_12 visible locations where points in noClaim are heavy ##naive_bayes found very important
  # # ps_car_13 visible areas where noClaim doesn't overlap much ##naive_bayes found very important
  # # ps_car_14 mayby minor areas where noClaim are heavy, but hard to tell ##naive_bayes found important
  # # ps_car_15 hard to see anything ##naive_bayes found very important
  # ####
  #   
  #   # train.unique <- lapply(ps.train.dt[,-c(1,2)],unique)
  # 
  # #---
  # # Deal with the -1 values in the interval variables!
  # #---
  # 
  # #---
  # # Covert categorical to factors!
  # #---
  #   # str(ps.train.dt)
  #   gc(verbose = TRUE)
  #   
  #   changeCols <- names(ps.train.dt)
  #   changeCols <- changeCols[(changeCols != "id")&(changeCols != "ps_car_13")&(changeCols != "ps_car_14")]
  #   ps.train.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
  #   # ps.test.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
  #   # str(ps.train.dt)
  #   
  # #---
  # # Scale the variables
  # #---
  
  featSelectTrain.dt <- performFeatSelection1(ps.train.dt)
  featSelectTest.dt  <- performFeatSelection1(ps.test.dt)
    str(featSelectTest.dt)
    
  #---
  # Create a validation set
  #---
    Train <- createDataPartition(ps.train.dt$targetChar,p=0.8,list=FALSE)
    train.dt <- ps.train.dt[Train,]    
    validate.dt <- ps.train.dt[-Train,]
    rm(Train)

    saveRDS(train.dt,"train.dt.RDS")
    saveRDS(validate.dt,"validate.dt.RDS")
    
    gc(verbose = TRUE)
  #---
  # Start Modelling!
  #---
    #---
    # Log Regression
    #---
      runLogReg <- function() {
        start_time <- Sys.time()
        set.seed(1000)
        
        # Set up resampling since we are imbalanced
        trainCtrl <- trainControl(method="repeatedcv", repeats = 5,
                                  #summaryFunction = twoClassSummary, 
                                  classProbs = TRUE,
                                  #savePredictions = TRUE,
                                  sampling = "smote")
        
        featVarLogReg.mod <- train(targetChar~., data=train.dt[,-c(1,2)], 
                                  method="glm", family="binomial", 
                                  metric = "ROC",
                                  trControl=trainCtrl)
        
        saveRDS(featVarLogReg.mod,file="featVarLogReg.mod.RDS")
        print("Logistic Regression Complete...")
        end_time <- Sys.time()
        write(paste("Log Reg took: ",end_time-start_time),file = "lot.txt",append = TRUE)
      }  
    #---
    # C5.0
    #---
      runC50 <- function() { 
        start_time <- Sys.time()
        set.seed(1000)
        # Set up resampling since we are imbalanced
        c50trainCtrl <- trainControl(method="repeatedcv", repeats = 5,
                                     #summaryFunction = twoClassSummary, 
                                     classProbs = TRUE,
                                     #savePredictions = TRUE,
                                     sampling = "smote")
        
        grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )
        
        featVarC50.mod <- train(targetChar~., data=train.dt[,-c(1,2)], 
                               tuneGrid=grid,method="C5.0",
                               # metric = "ROC",
                               trControl=c50trainCtrl)
        saveRDS(featVarC50.mod,file="featVarC50.mod.RDS")
        print("C5.0 Complete...")
        end_time <- Sys.time()
        write(paste("C5.0 took: ",end_time-start_time),file = "lot.txt",append = TRUE)
      }  
    #---
    # Naive Bayes
    #---
      runNB <- function () {  
        start_time <- Sys.time()
        set.seed(1000)
        
        NBtrainCtrl <- trainControl(method="repeatedcv", repeats = 5,
                                    #summaryFunction = twoClassSummary, 
                                    classProbs = TRUE,
                                    #savePredictions = TRUE,
                                    sampling = "smote")
        
        featVarNB <- train(targetChar~., data=train.dt[,-c(1,2)],
                          method = 'naive_bayes',
                          trControl=NBtrainCtrl)
        saveRDS(featVarNB,file="featVarNB.mod.RDS")
        print("Naive Bayes Complete...")
        end_time <- Sys.time()
        write(paste("Naive Bayes took: ",end_time-start_time),file = "lot.txt",append = TRUE)
      }  
    #---
    # nnet
    #---
      runNnet <- function () {  
        start_time <- Sys.time()
        set.seed(1000)
        
        NNtrainCtrl <- trainControl(method="repeatedcv", repeats = 5,
                                    #summaryFunction = twoClassSummary, 
                                    classProbs = TRUE,
                                    #savePredictions = TRUE,
                                    sampling = "smote")
        
        # NNgrid <- expand.grid(size=c(10),decay=c(0.1))
        NNgrid <- expand.grid(size=c(10))
        
        featVarNN <- train(targetChar~., data=train.dt[,-c(1,2)],
                          method = 'nnet',
                          # preProcess = c('scale'),
                          #tuneGrid = NNgrid,
                          trControl=NNtrainCtrl)
        saveRDS(featVarNN,file="featVarNN.mod.RDS")
        print("Neural Net Complete...")
        end_time <- Sys.time()
        write(paste("nnet took: ",end_time-start_time),file = "lot.txt",append = TRUE)
      }  
    
    #---
    # Try to run all the models, using try() in case something fails...
    #---
      print("Beginning...")
      try(runNB())
      gc(verbose = TRUE)
      try(runLogReg())
      gc(verbose = TRUE)
      try(runNnet())
      gc(verbose = TRUE)
      try(runC50())
      gc(verbose = TRUE)

  #---
  # Check the results
  #---
    train.dt <- readRDS(file = "train.dt.RDS")
    validate.dt <- readRDS(file = "validate.dt.RDS")
    
    featVarNB.mod <- readRDS(file = "featVarNB.mod.RDS")
    rm(featVarNB.mod)
    featVarLR.mod <- readRDS(file = "featVarLogReg.mod.RDS")
    
    this.mod <- featVarNB.mod
    this.mod <- featVarLR.mod
    
    summary(this.mod)
    this.mod$results
    varImp(this.mod)
    plot(varImp(this.mod))
    
    predictOut <- predict(this.mod, newdata = validate.dt)    
    confusionMatrix(predictOut,validate.dt$targetChar)
    
    predictProb <- predict(this.mod, newdata = validate.dt, type = 'prob')
    this.ROC <- roc(predictor=predictProb$claim, response = validate.dt$targetChar)
    plot(this.ROC,main='ROC')
    this.ROC$auc
  
  #---
  # Apply to actual test data...
  #---
    actualPred <- predict(this.mod, newdata = featSelectTest.dt, type = 'prob')
    actualOut <- data.table(id = featSelectTest.dt$id,target = actualPred$claim)
    # fwrite(actualOut,file="Samps_NaiveBayes_featSelct1.csv")
    fwrite(actualOut,file="Samps_LogReg_featSelct1.csv")
    