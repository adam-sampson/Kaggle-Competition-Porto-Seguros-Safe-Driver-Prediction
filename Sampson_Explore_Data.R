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
                "ggcorrplot",
                "tidyr")
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
  #### 
  # ps_car_13 correlates to ps_car_15
  # ps_car_04_cat correlates to ps_car_13 and ps_car_12
  # ps_reg_03 correlates to ps_reg_02 and ps_reg_01
  # ps_ind_14 correlates to ps_ind_11_bin and ps_ind_12_bin
  # ps_ind_16_bin negatively correlates to ps_ind_17_bin and ps_ind_18_bin
  ####
  
  # Bar charts for binary data
  # calculate new df with the percent of each bin
  
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
  
  #####
  # ps_ind_06_bin appears to have 5-10% relation to 0 or 1 target out
  # ps_ind_07_bin appears to have 5-10% relation to 0 or 1 target out
  # ps_ind_08_bin appears to have 1-3% relation to 0 or 1 target out
  # ps_ind_09_bin appears to have 1-2% relation to 0 or 1 target out
  # ps_ind_10_bin appears to have 0% relation to 0 or 1 target out
  # ps_ind_11_bin appears to have 0% relation to 0 or 1 target out
  # ps_ind_12_bin appears to have <1% relation to 0 or 1 target out
  # ps_ind_13_bin appears to have 0% relation to 0 or 1 target out
  # ps_ind_16_bin appears to have 3-6% relation to 0 or 1 target out
  # ps_ind_17_bin appears to have 2-5% relation to 0 or 1 target out
  # ps_ind_18_bin appears to have 1% relation to 0 or 1 target out
  # ps_calc_15_bin appears to have 0% relation to 0 or 1 target out
  # ps_calc_16_bin appears to have 1% relation to 0 or 1 target out
  # ps_calc_17_bin appears to have 0-1% relation to 0 or 1 target out
  # ps_calc_18_bin appears to have 0-1% relation to 0 or 1 target out
  # ps_calc_19_bin appears to have 1% relation to 0 or 1 target out
  # ps_calc_20_bin appears to have 1% relation to 0 or 1 target out
  ####
  
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
  
  temp <- calcPercentOfTotal(ps.train.dt,"ps_car_11_cat")
  temp <- temp %>% spread(target,percent)
  names(temp) <- c("value","x0","x1")
  temp <- temp %>% mutate(diff = x0-x1)
  max(temp$diff)
  min(temp$diff)
  mean(temp$diff)
  temp <- temp %>% mutate(keep = ifelse(diff <= 0,
                                        ifelse(abs(diff) > 0.005, -1, 0),
                                        ifelse(abs(diff) > 0.005, 1, 0)))
  rm(temp)
  
  ####
  # ps_ind_02_cat values 1 and -1/2 appear to have 2-3% relation, 3/4 <1% relation. 
  # ps_ind_04_cat values 1 and -1/2 appear to have 2-3% relation
  # ps_ind_05_cat values 0 and -1/2/4/5 appear to have 3-6% relation, 1/3/5 very little
  # ps_car_01_cat values 6/7 and -1/9/11 appear to have 3-6% relation, 0/1/5/8 less, and 1/2/3/10 very little
  # ps_car_02_cat values 0 and 1 appear to have 2-4% relation, -1 very little
  # ps_car_03_cat values -1 and 1 appear to have 3-6% relation, 0 <1% relation.
  # ps_car_04_cat values 0 and 1/2/3/4/5/6/7/8/9 appear to have 3-6% relation.
  # ps_car_05_cat values -1 and 1/2 appear to have 3-6%
  # ps_car_06_cat values 0/11 and 13/15/9/17, 1/4/14 and 2/5/7/8/10/11/12/16, 3/6 very little
  # ps_car_07_cat values -1/0 and 1 appear to have 1-2%
  # ps_car_08_cat values 0 and 1 appear to have 1-3%
  # ps_car_09_cat values 0 and 1 appear to have 2-4%, -1/2/3/4 very little
  # ps_cat_10_cat appears to have very little impact.
  # ps_car_11_cat needs a closer look (too many vars). Keep 41 and 104 as -1, keep 32,64,82,99,103 as 1, everything else 0.
  ####
  
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
  
  ####
  # ind_01 keep some 0/1 and 3/4/5/6/7, 2
  # ind_03 keep some 2/3/4 and 0/5/6/7/8, 1/9/10/11
  # ind_14 remove
  # ind_15 keep some 0/1/2/3/4/5/6/7 and 8/9/10/11/12/13
  # reg_01 keep some 1/2/3/4/5 and 0/7/8/9, 6
  # reg_02 keep some 0/1/2/3 and 5/6/7/8/9/1/11/12/13/14/15/16/17/18, 4
  # car_11 remove
  # calc_01 remove
  # calc_02 unsure
  # calc_03 unsure
  # calc_04 remove
  # calc_05 remove
  # calc_06 unsure
  # calc_07 unsure
  # calc_08 remove
  # calc_09 unsure
  # calc_10 unsure
  # calc_11 unsure
  # calc_12 remove
  # calc_13 remove
  # calc_14 unsure
  ####
  
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
  
  ####
  # ps_reg_03 minimal separation
  # ps_car_12 minimal separation
  # ps_car_13 small separation
  # ps_car_14 small separation
  # ps_car_15 minimal separation
  ####
  
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

  ####
  # ps_reg_03 visibal difference where 0 doesn't overlap
  # ps_car_12 visible locations where points in noClaim are heavy
  # ps_car_13 visible areas where noClaim doesn't overlap much
  # ps_car_14 mayby minor areas where noClaim are heavy, but hard to tell
  # ps_car_15 hard to see anything
  ####
  
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
  
  cleanupMissing(ps.train.dt,typeIndex.dt,0.05,-1)
  returnMissing(ps.train.dt,-1)
  head(ps.train.dt[get("ps_reg_03")==-1])
  head(ps.train.dt)
  head(ps.train.dt[get("ps_car_14")==-1])
  