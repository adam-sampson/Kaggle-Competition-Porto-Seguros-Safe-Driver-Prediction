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
              "naivebayes",
              "doParallel",
              "grid",
              "DMwR"
              )
loadPackages(packages)
rm(packages)

source("Functions.R")

set.seed(42)

#---
# Try modelling after Feature engineering
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
  
  #---
  # Cleanup Missing
  #---
    typeIndex.dt <- data.table()
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("binary"), colHeader = binary.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("categorical"), colHeader = categorical.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("ordinal"), colHeader = ordinal.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("interval"), colHeader = interval.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("target"), colHeader = target.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("id"), colHeader = c("id")))
    
    returnMissing(ps.train.dt,-1)
    
    cleanupMissing(ps.train.dt,typeIndex.dt,0.05,-1)
    returnMissing(ps.train.dt,-1)
    head(ps.train.dt[get("ps_reg_03")==-1])
    str(ps.train.dt)
    
    cleanupMissing(ps.test.dt,typeIndex.dt,0.05,-1)
    returnMissing(ps.test.dt,-1)
    str(ps.test.dt)
    
    rm(typeIndex.dt) 
    gc()
  
  #---
  # Feature Selection
  #---
    # select everything for comparison...
    featSelectTrain.dt <- ps.train.dt
    
    #feature selection option 2
     featSelectTrain.dt <- performFeatSelection4(ps.train.dt)
    # str(featSelectTrain.dt)
    # summary(featSelectTrain.dt)
     featSelectTest.dt  <- performFeatSelection4(ps.test.dt)
    # str(featSelectTest.dt)
    
    #feature selection option 3
    # featSelectTrain.dt <- performFeatSelection3(ps.train.dt)
    # featSelectTest.dt  <- performFeatSelection3(ps.test.dt)
    
    #feature selection option 4
    # featSelectTrain.dt <- performFeatSelection4(ps.train.dt)
    # featSelectTest.dt  <- performFeatSelection4(ps.test.dt)
    
    # What if we remove the huge ps_car_11?
    # featSelectTest.dt <- featSelectTest.dt %>% select(-ps_car_11_cat)
    # featSelectTrain.dt <- featSelectTrain.dt %>% select(-ps_car_11_cat)
    
    # Convert to factors
    categorical.var <- names(featSelectTrain.dt) %>% str_subset(".+_cat")
    binary.var <- names(featSelectTrain.dt) %>% str_subset(".+_bin")
    miss.var <- names(featSelectTrain.dt) %>% str_subset(".+_miss")
    changeCols <- c(binary.var,categorical.var,target.var,miss.var)
    featSelectTrain.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
    str(featSelectTrain.dt) 
    
    changeCols <- str_subset(changeCols,"^(?!target).*$")
    featSelectTest.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
    str(featSelectTest.dt) 
  #---
  # sample for smaller set...for testing purposes only
  #---
    set.seed(1000)
    featSelectTrain.dt <- featSelectTrain.dt %>% sample_n(20000)
    confusionMatrix(featSelectTrain.dt$targetChar,featSelectTrain.dt$targetChar)

    
     
    
  #---
  # Create a validation set
  #---
    Train <- createDataPartition(featSelectTrain.dt$targetChar,p=0.8,list=FALSE)
    train.dt <- featSelectTrain.dt[Train,]    
    validate.dt <- featSelectTrain.dt[-Train,]
    rm(Train)
    rm(featSelectTrain.dt)
    # rm(ps.train.dt)
    # rm(ps.test.dt)
    
    # saveRDS(train.dt,"train.dt.RDS")
    # saveRDS(validate.dt,"validate.dt.RDS")
    
    gc(verbose = TRUE)
  #---
  # Start Modelling!
  #---
    #---
    # Log Regression
    #---
      runLogReg <- function() {
        start_time <- Sys.time()
        write(paste("Log Reg started at: ",start_time),file = "log.txt",append = TRUE)
        set.seed(1000)
        
        # Set up resampling since we are imbalanced
        # trainCtrl <- trainControl(method="repeatedcv", repeats = 5,
        # trainCtrl <- trainControl(method="cv",
        trainCtrl <- trainControl(method="none",
                                  #summaryFunction = twoClassSummary, 
                                  classProbs = TRUE,
                                  savePredictions = FALSE,
                                  sampling = "smote")
        
        featVarLogReg.mod <- train(targetChar~., data=train.dt[,-c(1,2)], 
                                  method="glm", family="binomial", 
                                  metric = "ROC",
                                  trControl=trainCtrl)
        
        saveRDS(featVarLogReg.mod,file="featVarLogReg.mod.RDS")
        print("Logistic Regression Complete...")
        end_time <- Sys.time()
        print(paste("Log Reg took: ",end_time-start_time))
        write(paste("Log Reg ended at: ",end_time),file = "log.txt",append = TRUE)
        write(paste("Log Reg took: ",end_time-start_time),file = "log.txt",append = TRUE)
        gc(verbose = TRUE)
        return(featVarLogReg.mod)
      }  
    #---
    # C5.0
    #---
      runC50 <- function() { 
        start_time <- Sys.time()
        write(paste("C5.0 started at: ",start_time),file = "log.txt",append = TRUE)
        set.seed(1000)
        # Set up resampling since we are imbalanced
        # c50trainCtrl <- trainControl(method="cv",
        c50trainCtrl <- trainControl(#method="none",
                                     #<- trainControl(method="repeatedcv", repeats = 5,
                                     #summaryFunction = twoClassSummary, 
                                     classProbs = TRUE,
                                     #savePredictions = TRUE,
                                     sampling = "smote")
        
        grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )
        
        featVarC50.mod <- train(targetChar~., data=train.dt[,-c(1,2)], 
                               tuneGrid=grid,
                               method="C5.0",
                               metric = "ROC",
                               trControl=c50trainCtrl)
        saveRDS(featVarC50.mod,file="featVarC50.mod.RDS")
        print("C5.0 Complete...")
        end_time <- Sys.time()
        print(paste("C5.0 took: ",end_time-start_time))
        write(paste("C5.0 ended at: ",end_time),file = "log.txt",append = TRUE)
        write(paste("C5.0 took: ",end_time-start_time),file = "log.txt",append = TRUE)
        return(featVarC50.mod)
      }  
    #---
    # Naive Bayes
    #---
      runNB <- function () {  
        start_time <- Sys.time()
        write(paste("Naive Bayes started at: ",start_time),file = "log.txt",append = TRUE)
        set.seed(1000)
        
        # NBtrainCtrl <- trainControl(method="cv",
        NBtrainCtrl <- trainControl(#method="none",
                                    #<- trainControl(method="repeatedcv", repeats = 5,
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
        print(paste("Naive Bayes took: ",end_time-start_time))
        write(paste("Naive Bayes ended at: ",end_time),file = "log.txt",append = TRUE)
        write(paste("Naive Bayes took: ",end_time-start_time),file = "log.txt",append = TRUE)
        return(featVarNB)
      }  
    #---
    # nnet
    #---
      runNnet <- function () {  
        start_time <- Sys.time()
        write(paste("nnet started at: ",start_time),file = "log.txt",append = TRUE)
        set.seed(1000)
        
        # NNtrainCtrl <- trainControl(method="cv",
        NNtrainCtrl <- trainControl(method="none",
                                    #<- trainControl(method="repeatedcv", repeats = 5,
                                    #summaryFunction = twoClassSummary, 
                                    classProbs = TRUE,
                                    #savePredictions = TRUE,
                                    sampling = "smote")
        
        # NNgrid <- expand.grid(size=c(10),decay=c(0.1))
        NNgrid <- expand.grid(size=c(10))
        
        featVarNN <- train(targetChar~., data=train.dt[,-c(1,2)],
                          method = 'nnet',
                          # preProcess = c('scale'),
                          # tuneGrid = NNgrid,
                          trControl=NNtrainCtrl)
        saveRDS(featVarNN,file="featVarNN.mod.RDS")
        print("Neural Net Complete...")
        end_time <- Sys.time()
        print(paste("NNet took: ",end_time-start_time))
        write(paste("nnet ended at: ",end_time),file = "log.txt",append = TRUE)
        write(paste("nnet took: ",end_time-start_time),file = "log.txt",append = TRUE)
        return(featVarNN)
      }  
    
    #---
    # KNN
    #---
      runKNN <- function () {  
        start_time <- Sys.time()
        set.seed(1000)
        
        # KNNtrainCtrl <- trainControl(method="cv",
        KNNtrainCtrl <- trainControl(#method="none",
                                    #<- trainControl(method="repeatedcv", repeats = 5,
                                    #summaryFunction = twoClassSummary, 
                                    classProbs = TRUE,
                                    #savePredictions = TRUE,
                                    sampling = "smote")
        
        # NNgrid <- expand.grid(size=c(10),decay=c(0.1))
        #KNNgrid <- expand.grid(size=c(10))
        
        featVarKNN <- train(targetChar~., data=train.dt[,-c(1,2)],
                           method = 'nnet',
                           trControl=KNNtrainCtrl,
                           preProcess = c("center", "scale"),
                           tuneLength = 20)
        saveRDS(featVarKNN,file="featVarKNN.mod.RDS")
        print("KNN Complete...")
        end_time <- Sys.time()
        print(paste("KNN took: ",end_time-start_time))
        write(paste("KNN took: ",end_time-start_time),file = "log.txt",append = TRUE)
        return(featVarKNN)
      } 
    #---
    # Try to run all the models, using try() in case something fails...
    #---
      
      #ppCL <- makeCluster(3)
      gc(verbose=TRUE)
      #gc()
      print(paste("Beginning at",Sys.time()))
      
        try(NB.mod <- runNB())
        gc(verbose = TRUE)
        
        try(NN.mod <- runNnet())
        gc(verbose = TRUE)
        
        try(c50.mod <- runC50())
        gc(verbose = TRUE)
        
        try(runKNN())
        gc(verbose = TRUE)
        
        try(LogReg.mod <- runLogReg())
        gc(verbose = TRUE)
      

  #---
  # Check the results
  #---
    # train.dt <- readRDS(file = "train.dt.RDS")
    # validate.dt <- readRDS(file = "validate.dt.RDS")
    
    NB.mod <- readRDS(file = "featVarNB.mod.RDS")
    rm(featVarNB.mod)
    gc(verbose = TRUE)

    # featVarLR.mod <- readRDS(file = "featVarLogReg.mod2.RDS")
    # rm(featVarLR.mod)
    # gc(verbose = TRUE)
    # 
    NN.mod <- readRDS(file = "featVarNN.mod.RDS")
    rm(featVarNN.mod)
    gc(verbose = TRUE)

    c50.mod <- readRDS(file = "featVarC50.mod.RDS")
        
    LogReg.mod <- readRDS(file = "featVarLogReg.mod2.RDS")
    
    rm(this.mod)
    gc(verbose=TRUE)
    
    # this.mod <- featVarNB.mod
    # this.mod <- featVarLR.mod
    # this.mod <- featVarNN.mod
    
    this.mod <- LogReg.mod
    this.mod <- NB.mod
    this.mod <- NN.mod
    this.mod <- c50.mod
    
    # rm(featVarLR.mod)
    gc(verbose = TRUE)
    # .rs.restartR()
    
    summary(this.mod)
    summary.mod <- summary(this.mod)
    # capture.output(summary.mod,file = "LR_feat4full_modelSummary.txt")
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
    gc()
    actualOut <- data.table(id = featSelectTest.dt$id,target = actualPred$claim)
    # fwrite(actualOut,file="Samps_NaiveBayes_featSelct1.csv")
    fwrite(actualOut,file="Samps_LogReg_featSelct4.csv")
    # fwrite(actualOut,file="Samps_NN_featSelct2.csv")
    # fwrite(actualOut,file="Samps_NB_featSelct2.csv")
    rm(actualPred)
    rm(actualOut)
    gc()
    