##########
# Kaggle Competition
# Porto Seguro's Safe Driver Prediction
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43282
# Author: Adam Sampson
# 
# Goal: Predict the probability that a driver will initiate an auto claim in the next year.
# Clarification: Maximize false negatives, and maximize false positives
##########

#---
# Data review:
# The data provided is highly imbalanced data where the target variable occurs very rarely.
# The target has a value of 0 for about 96.3% of the samples. Therefore a model guessing 0 
# for 100% of samples would be 96.3% accurate, but with a high false negative rate. Therefore,
# the goal of this data is to decrease the false negative rate without over-inflating the false
# positive rate. 
#---

  source("LoadPackages.R")
  
  packages <- c("data.table")
  loadPackages(packages)
  rm(packages)
  
  source("Functions.R")

#---
# Import Data from files as data.table
#---
  ps.train.dt <- fread("train.csv")
  ps.test.dt <- fread("test.csv")
  
  View(head(ps.train.dt, n = 20L))

#---
# This data is heavily imbalanced. Let's split the training data into positives and negatives 
# to help review. data.table makes this more memory efficient.
#---
  pos.ps.train.dt <- ps.train.dt[target == 1,]  
  neg.ps.train.dt <- ps.train.dt[target == 0,]
  
#---
# Review data and clean up classes of variables
#---
  str(ps.train.dt)
  summary(ps.train.dt)
  
  # A lot of this data is categorical, even if the answers are integers. Need to identify which
  # variables need to be treated as categorical.
  # QUESTION: Can we use factors to handle categorical values?
  
  # Indentify how many values are possible per column
  unique.train <- lapply(ps.train.dt,unique)
    # ID is obviously not categorical. 
    # ps_reg_03 has 5000 possible decimal values -> not likely categorical
    # ps_car_13 has >70k possilbe decimal values -> highly unlikely categorical

  unique.test <- lapply(ps.test.dt,unique)
  
  # Find features which have values that are not in the train set but are in the test set
  valuesOnlyInTest <- checkForValuesInBoth(unique.test,unique.train)
  print(valuesOnlyInTest)
    # This results in id (obviously), ps_reg_03, ps_car_12, ps_car_13, ps_car_14, ps_calc_08, 
    # ps_calc_11, ps_calc_13, and ps_calc_14 having values in test set that aren't in the training set
  
  # Cleanup exploratory data
  rm(unique.train)
  rm(unique.test)
  rm(valuesOnlyInTest)
  gc(verbose = TRUE)
  
