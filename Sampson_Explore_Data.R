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
                "dummies",
                "neuralnet",
                "stringr",
                "ggplot2",
                "caret",
                "scales",
                "ggthemes",
                "ggExtra",
                "ggcorrplot")
  loadPackages(packages)
  rm(packages)
  
  source("Functions.R")

#---
# Import Data from files as data.table
#---
  ps.train.dt <- fread("train.csv")
  ps.test.dt <- fread("test.csv")

#---
# Review data and clean up classes of variables
#---
  #---
  # Determine type of data categorical, ordinal, binary, integer
  #---
    View(head(ps.train.dt, n = 20L))
    train.unique <- lapply(ps.train.dt[,-c(1,2)],unique)
    test.unique <- lapply(ps.test.dt[,-1],unique)
    
    # Data is divided into id, target, _ind, _reg, _car, _calc
    # Data is class of continuous(not marked), ordinal(not marked), _bin, _cat
    
    target.var <- c("target")
    categorical.var <- names(ps.train.dt) %>% str_subset(".+_cat")
    binary.var <- names(ps.train.dt) %>% str_subset(".+_bin")
    other.var <- ps.train.dt %>% select(-id,-target, -one_of(categorical.var), -one_of(binary.var)) %>% names()
    
    # Review the remaing to estimate whether they appear to be ordinal or interval
    View(head(ps.train.dt %>% select(one_of(other.var)),20))
    unique.train.other <- lapply(ps.train.dt %>% select(one_of(other.var)),unique)
      unique.train.other
      lapply(unique.train.other,summary)
      lapply(unique.train.other,length)
      summary(ps.train.dt %>% select(one_of(other.var)))
      # See whether any of these variables are missing values in the ps.test.dt
      valuesOnlyInTest <- checkForValuesInBoth(
        lapply(ps.train.dt %>% select(one_of(other.var)),unique),
        lapply(ps.test.dt %>% select(one_of(other.var)),unique))
      # Review what we found while reviewing...
        # ps_ind_01 are integers between 1 and 7
        # ps_ind_03 are integers between 1 and 11
        # ps_ind_14 are integers between 1 and 4
        # ps_ind_15 are integers between 1 and 13
        # ps_reg_01 are decimals between 0.0 and 0.9 (but only one digit)
        # ps_reg_02 are decimals between 0.0 and 1.7 (but only one digit)
        # ps_reg_03 are decimals with many values including -1 values to indicate unknown and outliers above 1
        # ps_car_11 are integers between 0 to 3 with some -1 values to indicate unknown
        # ps_car_12 are decimals with many values with some -1 values to indicate unknown and some outliders above 1
        # ps_car_13 are decimals with many values and some outliers above 3.7
        # ps_car_14 are decimals with many values with some -1 values to indicate unknown and limited range
        # ps_car_15 are decimals with only a few values between 0 and 3.741657. Could this be a few categories 
            # that are transformed?
        # ps_calc_01 are decimals between 0.0 and 0.9 (but only one digit)
        # ps_calc_02 are decimals between 0.0 and 0.9 (but only one digit)
        # ps_calc_03 are decimals between 0.0 and 0.9 (but only one digit)
        # ps_calc_04 are integers between 0 and 5
        # ps_calc_05 are integers between 0 and 6
        # ps_calc_06 are integers between 0 and 10
        # ps_calc_07 are integers between 0 and 9
        # ps_calc_08 are integers between 2 and 12
        # ps_calc_09 are integers between 0 and 7
        # ps_calc_10 are integers between 0 and 25
        # ps_calc_11 are integers between 0 and 19
        # ps_calc_12 are integers between 0 and 10
        # ps_calc_13 are integers between 0 and 13
        # ps_calc_14 are integers between 0 and 23
    
    # Make an educated guess which variables appear to be ordinal
    ordinal.var <- ps.train.dt %>% select(other.var) %>% select(1,2,3,4,5,6,8,13:26) %>% names()
    interval.var <- ps.train.dt %>% select(-id,-one_of(target.var),-one_of(categorical.var),
                                          -one_of(binary.var),-one_of(ordinal.var)) %>% names()
    rm(other.var)
    
  #---
  # Convert variables to different data types.
  #---
    str(ps.train.dt)
    changeCols <- c(binary.var,categorical.var,interval.var,ordinal.var,target.var)
    ps.train.dt[,(changeCols) := lapply(.SD,as.numeric), .SDcols = changeCols]
    str(ps.train.dt)
    
    changeCols <- target.var
    
    # ps.train.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
    
#---
# Visualize data for patterns and feature selection
#---
  corr <- round(cor(ps.train.dt))
  ggcorrplot(corr, hc.order=TRUE,
             type="upper",
             lab=TRUE,
             lab_size=3,
             method="circle")
    
  # Bar charts for binary data
  # calculate new df with the percent of each bin
  calcPercentOfTotal <- function(in.dt,colName) {
    percent.df <- data.frame()
    totTargZero <- length(in.dt[target==0,target])
    totTargOne  <- length(in.dt[target==1,target])
    for(vals in unique(select(in.dt,colName)[[1]])) {
      temp.dt <- in.dt[get(colName) == vals]
      #valCount <- length(temp.dt[,get(colName)])
      percentPos <- length(temp.dt[target == 1,get(colName)]) / totTargOne
      percentNeg <- length(temp.dt[target == 0,get(colName)]) / totTargZero
      percent.df <- percent.df %>% rbind(data.frame(target = c(1,0),
                                                    value = c(vals,vals), 
                                                    percent = c(percentPos,percentNeg)))
    }
    return(percent.df)
  }
  
  calcPercentOfTotal(ps.train.dt,"ps_ind_06_bin")
  
  for (i in binary.var) {
    plot <- ggplot(calcPercentOfTotal(ps.train.dt,i)) +
      geom_bar(aes(x=value,
                   y=percent,
                   fill=factor(target)),
               position="dodge",
               stat="identity") +
      ggtitle(i)
    print(plot)
    readline(prompt = "Press enter to view next plot.")
  }
  
  for (i in categorical.var) {
    plot <- ggplot(calcPercentOfTotal(ps.train.dt,i)) +
      geom_bar(aes(x=value,
                   y=percent,
                   fill=factor(target)),
               position="dodge",
               stat="identity") +
      ggtitle(i)
    print(plot)
    readline(prompt = "Press enter to view next plot.")
  }
  
  for (i in ordinal.var) {
    plot <- ggplot(calcPercentOfTotal(ps.train.dt,i)) +
      geom_bar(aes(x=value,
                   y=percent,
                   fill=factor(target)),
               position="dodge",
               stat="identity") +
      ggtitle(i)
    print(plot)
    readline(prompt = "Press enter to view next plot.")
  }
  
  ## For density plots
  
  for (i in ordinal.var) {
    plot <- ggplot(ps.train.dt) +
      geom_density(aes_string(x=i,fill="factor(target)"),alpha=0.6) +
      ggtitle(i)
    print(plot)
    readline(prompt = "Press enter to view next plot.")
  }
  
  for (i in interval.var) {
    plot <- ggplot(ps.train.dt) +
      geom_density(aes_string(x=i,fill="factor(target)"),alpha=0.6) +
      ggtitle(i)
    print(plot)
    readline(prompt = "Press enter to view next plot.")
  }
  
  for (i in ordinal.var) {
    plot <- ggplot(ps.train.dt) +
      geom_count(aes_string(x=i,y="factor(target)")) +
      ggtitle(i)
    # ggMarginal(plot, type="histogram",fill="transparent")
    print(plot)
    readline(prompt = "Press enter to view next plot.")
  }
  
  for (i in interval.var) {
    plot <- ggplot(ps.train.dt) +
      geom_count(aes_string(x=i,y="factor(target)")) +
      ggtitle(i)
    # ggMarginal(plot, type="histogram",fill="transparent")
    print(plot)
    readline(prompt = "Press enter to view next plot.")
  }
  
  plot <- ggplot(ps.train.dt) +
    geom_count(aes_string(x="ps_reg_03",y="factor(target)"))
  ggMarginal(plot, type="histogram",fill="transparent")
  print(plot)  

#---
# Consider what to do about -1 values.
#---
  # count the number of values that are -1 in each variable
  returnMissing(ps.train.dt,-1)
  
  ## One option is to run a tree, each branch being whether -1 is a value in a specific column
  
  ## One option is to convert the -1 to something else
  
  ## One option (for some categorical, binary, or ordinal) is to treat -1 as a category. 
  ## Only ps_reg_03 (18%) would have a problem with this method
  
  ## One option is to make a categorical variable for each column that has a missing value
  ## and then deal with the -1 in the main column with median, mode, or mean.
  typeIndex.dt <- data.table()
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("binary"), colHeader = binary.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("categorical"), colHeader = categorical.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("ordinal"), colHeader = ordinal.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("interval"), colHeader = interval.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("target"), colHeader = target.var))
    typeIndex.dt <- rbind(typeIndex.dt,data.table(type = c("id"), colHeader = c("id")))
  
  cleanupMissing <- function(in.dt,classIndex.dt,cutoffPct,missingVal) {
    colWithMissing <- returnMissing(in.dt,missingVal)
    for(col in names(colWithMissing)) {
      # if the percentage is greater than cutoffPct create a dummy binary variable
      if(as.numeric(colWithMissing[[col]][1]) > cutoffPct) {
        print("Over cutoff percent")
      }
      # for all variables with missing, decide whether to use mean, median, or mode to cleanup variable
      if(classIndex.dt[colHeader == col][[1]] == "interval"){
        print("Interval Variable Detected")
      } else if(classIndex.dt[colHeader == col][[1]] == "ordinal"){
        print("Ordinal Variable Detected")
      } else if(classIndex.dt[colHeader == col][[1]] == "categorical"){
        print("Categorical Variable Detected")
      } else if(classIndex.dt[colHeader == col][[1]] == "binary"){
        print("Binary Variable Detected")
      }
      
    }
  }
  