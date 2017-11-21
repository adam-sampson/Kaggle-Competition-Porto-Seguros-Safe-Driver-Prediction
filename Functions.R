##########
# File to load functions cleanly outside main program(s)
##########

# Note: use gc() to clean up memory...
# Note: use gc(verbose=TRUE) to clean up more
# Note: use memory.size() to see how much memory is being used...

checkForValuesInBoth <- function(sourceList, checkList) {
  outList <- list()
  for(feature in names(sourceList)) {
    # print(paste("Feature:",feature))
    tempCatch <- NULL
    logical.check <- sourceList[[feature]] %in% checkList[[feature]]
    tempCatch <- sourceList[[feature]][!logical.check]
    if(length(tempCatch) != 0) {  
      outList[[feature]] <- tempCatch
    }
  }
  return(outList)
}

returnMissing <- function(source.dt,missingChar) {
  outList <- list()
  totallength <- length(source.dt[[1]])
  for(feature in names(source.dt)) {
    tempCatch <- NULL
    tempCatch <- length(source.dt[get(feature)==missingChar,get(feature)])
    if(tempCatch != 0) {
      outList[[feature]] <- c(round(tempCatch/totallength,digits = 5),paste0(tempCatch," (",round(100*tempCatch/totallength,digits = 3),"%) of values are ",as.character(missingChar)))
    }
  }
  return(outList)
}

Mode = function(x){ 
  ta = table(x)
  tam = max(ta)
  if (all(ta == tam))
    mod = NA
  else
    if(is.numeric(x))
      mod = as.numeric(names(ta)[ta == tam])
  else
    mod = names(ta)[ta == tam]
  return(mod)
}

# Knowing that -1 values are missing values...clean them up
cleanupMissing <- function(in.dt,classIndex.dt,cutoffPct,missingVal) {
  colWithMissing <- returnMissing(in.dt,missingVal)
  for(col in names(colWithMissing)) {
    # if the percentage is greater than cutoffPct create a dummy binary variable
    if(as.numeric(colWithMissing[[col]][1]) > cutoffPct) {
      print("Over cutoff percent")
      newCol <- paste0(col,"_miss")
      in.dt[,(newCol) := ifelse(.SD==missingVal,1,0),.SDcols = (col)]
    }
    
    # for all variables with missing, decide whether to use mean, median, or mode to cleanup variable
    if(classIndex.dt[colHeader == col][[1]] == "interval"){
      print("Interval Variable Detected")
      # Use the mean for interval
      # newCol <- paste0(col,"_meanfix")
      tempMean <- mean(in.dt[get(col)!=missingVal,get(col)])
      in.dt[get(col)==missingVal,(col) := tempMean]
      #in.dt[get(col)!=missingVal,(col) := .SD, .SDcols = (col)]
    } else if(classIndex.dt[colHeader == col][[1]] == "ordinal"){
      print("Ordinal Variable Detected")
      tempMode <- Mode(in.dt[get(col)!=missingVal,get(col)])
      in.dt[get(col)==missingVal,(col) := tempMode]
    } else if(classIndex.dt[colHeader == col][[1]] == "categorical"){
      print("Categorical Variable Detected")
      tempMode <- Mode(in.dt[get(col)!=missingVal,get(col)])
      in.dt[get(col)==missingVal,(col) := tempMode]
    } else if(classIndex.dt[colHeader == col][[1]] == "binary"){
      print("Binary Variable Detected")
      tempMode <- Mode(in.dt[get(col)!=missingVal,get(col)])
      in.dt[get(col)==missingVal,(col) := tempMode]
    } else {
      
    }
    
  }
}

opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
} 

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

performFeatSelection1 <- function(in.dt) {
  ####
  # Feature Engineering!
  #### 
  # ps_car_13 correlates to ps_car_15(remove)
  # ps_car_04_cat(remove) correlates to ps_car_13 and ps_car_12(remove)
  # ps_reg_03(remove) correlates to ps_reg_02 and ps_reg_01(remove)
  # ps_ind_14(remove) correlates to ps_ind_11_bin(remove) and ps_ind_12_bin
  # ps_ind_16_bin(remove) negatively correlates to ps_ind_17_bin and ps_ind_18_bin(remove)
  ####
  in.dt <- in.dt %>% select(-ps_car_15, 
                                        -ps_car_04_cat,
                                        -ps_car_12,
                                        -ps_reg_03,
                                        -ps_reg_01,
                                        -ps_ind_14,
                                        -ps_ind_11_bin,
                                        -ps_ind_16_bin,
                                        -ps_ind_18_bin)
  
  #####
  # ps_ind_06_bin appears to have 5-10% relation to 0 or 1 target out ##naive_bayes found important
  # ps_ind_07_bin appears to have 5-10% relation to 0 or 1 target out ##naive_bayes found important
  # ps_ind_08_bin appears to have 1-3% relation to 0 or 1 target out (remove)
  # ps_ind_09_bin appears to have 1-2% relation to 0 or 1 target out (remove)
  # ps_ind_10_bin appears to have 0% relation to 0 or 1 target out (remove)
  # ps_ind_11_bin appears to have 0% relation to 0 or 1 target out (remove)
  # ps_ind_12_bin appears to have <1% relation to 0 or 1 target out (remove)
  # ps_ind_13_bin appears to have 0% relation to 0 or 1 target out (remove)
  # ps_ind_16_bin appears to have 3-6% relation to 0 or 1 target out
  # ps_ind_17_bin appears to have 2-5% relation to 0 or 1 target out
  # ps_ind_18_bin appears to have 1% relation to 0 or 1 target out (remove)
  # ps_calc_15_bin appears to have 0% relation to 0 or 1 target out (remove)
  # ps_calc_16_bin appears to have 1% relation to 0 or 1 target out (remove)
  # ps_calc_17_bin appears to have 0-1% relation to 0 or 1 target out ##naive_bayes found important (remove)
  # ps_calc_18_bin appears to have 0-1% relation to 0 or 1 target out (remove)
  # ps_calc_19_bin appears to have 1% relation to 0 or 1 target out (remove)
  # ps_calc_20_bin appears to have 1% relation to 0 or 1 target out (remove)
  ####
  in.dt <- in.dt %>% select(-ps_ind_08_bin,-ps_ind_09_bin,-ps_ind_10_bin,-ps_ind_12_bin,
                                        -ps_ind_13_bin,-ps_calc_15_bin,-ps_calc_16_bin,-ps_calc_17_bin,
                                        -ps_calc_18_bin,-ps_calc_19_bin,-ps_calc_20_bin)
  
  
  ####
  # ps_ind_02_cat values 1 and -1/2 appear to have 2-3% relation, 3/4 <1% relation. 
  # ps_ind_04_cat values 1 and -1/2 appear to have 2-3% relation
  # ps_ind_05_cat values 0 and -1/2/4/5 appear to have 3-6% relation, 1/3/5 very little
  # ps_car_01_cat values 6/7 and -1/9/11 appear to have 3-6% relation, 0/1/5/8 less, and 1/2/3/10 very little ##naive_bayes found important
  
  # ps_car_02_cat values 0 and 1 appear to have 2-4% relation, -1 very little (remove)
  # ps_car_03_cat values -1 and 1 appear to have 3-6% relation, 0 <1% relation.
  # ps_car_04_cat values 0 and 1/2/3/4/5/6/7/8/9 appear to have 3-6% relation. (removed earlier in correlation)
  # ps_car_05_cat values -1 and 1/2 appear to have 3-6%
  # ps_car_06_cat values 0/11 and 13/15/9/17, 1/4/14 and 2/5/7/8/10/11/12/16, 3/6 very little
  
  # ps_car_07_cat values -1/0 and 1 appear to have 1-2% (remove)
  # ps_car_08_cat values 0 and 1 appear to have 1-3% (remove)
  # ps_car_09_cat values 0 and 1 appear to have 2-4%, -1/2/3/4 very little
  # ps_car_10_cat appears to have very little impact. (remove)
  # ps_car_11_cat needs a closer look (too many vars). Keep 41 and 104 as -1, keep 32,64,82,99,103 as -2, everything else 0.
  ####
  in.dt <- in.dt %>% select(-ps_car_07_cat,-ps_car_08_cat,-ps_car_10_cat, -ps_car_02_cat)
  in.dt[ps_ind_02_cat == -1, ps_ind_02_cat := 2]
  in.dt[ps_ind_02_cat == 4, ps_ind_02_cat := 3]
  in.dt[ps_ind_04_cat == -1, ps_ind_04_cat := 2]
  in.dt[(ps_ind_05_cat == -1)|(ps_ind_05_cat == 2)|(ps_ind_05_cat == 4)|(ps_ind_05_cat == 5), ps_ind_05_cat := 2]
  in.dt[(ps_ind_05_cat == 1)|(ps_ind_05_cat == 3)|(ps_ind_05_cat == 5), ps_ind_05_cat := 1]
  in.dt[(ps_ind_05_cat != 2)&(ps_ind_05_cat != 1),ps_ind_05_cat:=0]
  in.dt[(ps_car_01_cat==7),ps_car_01_cat:=6]
  in.dt[(ps_car_01_cat==-1)|(ps_car_01_cat ==9)|(ps_car_01_cat==11),ps_car_01_cat:=-1]
  in.dt[(ps_car_01_cat != -1)&(ps_car_01_cat != 6),ps_car_01_cat:=0]
  in.dt[(ps_car_05_cat == 2),ps_car_05_cat := 1]
  in.dt[(ps_car_06_cat == 11),ps_car_06_cat := 0]
  in.dt[(ps_car_06_cat == 9)|(ps_car_06_cat == 13)|(ps_car_06_cat == 15)|(ps_car_06_cat == 17),ps_car_06_cat:=9]
  in.dt[(ps_car_06_cat == 1)|(ps_car_06_cat == 4)|(ps_car_06_cat == 14),ps_car_06_cat:=1]
  in.dt[(ps_car_06_cat != 1)&(ps_car_06_cat != 9), ps_car_06_cat:=2]
  in.dt[(ps_car_09_cat != 0)&(ps_car_09_cat != 1),ps_car_09_cat := 2]
  in.dt[(ps_car_11_cat == 41)|(ps_car_11_cat == 104),ps_car_11_cat:=-1]
  in.dt[(ps_car_11_cat == 32)|(ps_car_11_cat == 64)|(ps_car_11_cat == 82)|(ps_car_11_cat == 99)|(ps_car_11_cat == 103),ps_car_11_cat := -2]
  in.dt[(ps_car_11_cat != -1)&(ps_car_11_cat != -2),ps_car_11_cat := 0]
  
  
  ####
  # ordinal
  # ind_01 keep some 0/1 and 3/4/5/6/7, 2
  # ind_03 keep some 2/3/4 and 0/5/6/7/8, 1/9/10/11
  # ind_14 remove
  # ind_15 keep some 0/1/2/3/4/5/6/7 and 8/9/10/11/12/13
  # reg_01 keep some 1/2/3/4/5 and 0/7/8/9, 6 ##naive_bayes found important (removed in correlation)
  # reg_02 keep some 0/1/2/3 and 5/6/7/8/9/1/11/12/13/14/15/16/17/18, 4 ##naive_bayes found important
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
  in.dt <- in.dt %>% select(-ps_car_11,-ps_calc_01,-ps_calc_02,-ps_calc_03,-ps_calc_04,
                                        -ps_calc_05,-ps_calc_06,-ps_calc_07,-ps_calc_08,-ps_calc_09,
                                        -ps_calc_10,-ps_calc_11,-ps_calc_12,-ps_calc_13,-ps_calc_14)
  in.dt[ps_ind_01 == 1,ps_ind_01 := 0]
  in.dt[(ps_ind_01 == 3)|(ps_ind_01 == 4)|(ps_ind_01 == 5)|(ps_ind_01 == 6)|(ps_ind_01 == 7)|(ps_ind_01 == 8),ps_ind_01 := 3]
  in.dt[(ps_ind_03 == 3)|(ps_ind_03 == 4),ps_ind_03 := 2]
  in.dt[(ps_ind_03 == 5)|(ps_ind_03 == 6)|(ps_ind_03 == 7)|(ps_ind_03 == 8),ps_ind_03 := 0]
  in.dt[(ps_ind_03 != 2)&(ps_ind_03 != 0), ps_ind_03 := 1]
  in.dt[(ps_ind_15 == 0)|(ps_ind_15 == 1)|(ps_ind_15 == 2)|(ps_ind_15 == 3)|(ps_ind_15 == 4)|(ps_ind_15 == 5)|(ps_ind_15 == 6)|(ps_ind_15 == 7),ps_ind_15 := 0]
  in.dt[(ps_ind_15 != 0),ps_ind_15 := 8]
  in.dt[(ps_reg_02 == 1)|(ps_reg_02 == 2)|(ps_reg_02 == 3),ps_reg_02 := 0]
  in.dt[(ps_reg_02 != 0)&(ps_reg_02 != 4), ps_reg_02 := 5]
  
  ####
  # ps_reg_03 visibal difference where 0 doesn't overlap ##naive_bayes found important
  # ps_car_12 visible locations where points in noClaim are heavy ##naive_bayes found very important
  # ps_car_13 visible areas where noClaim doesn't overlap much ##naive_bayes found very important
  # ps_car_14 mayby minor areas where noClaim are heavy, but hard to tell ##naive_bayes found important
  # ps_car_15 hard to see anything ##naive_bayes found very important
  ####
  
  # train.unique <- lapply(in.dt[,-c(1,2)],unique)
  
  #---
  # Deal with the -1 values in the interval variables!
  #---
  
  #---
  # Covert categorical to factors!
  #---
  # str(in.dt)
  gc(verbose = TRUE)
  
  changeCols <- names(in.dt)
  changeCols <- changeCols[(changeCols != "id")&(changeCols != "ps_car_13")&(changeCols != "ps_car_14")]
  in.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
  # ps.test.dt[,(changeCols) := lapply(.SD,as.factor), .SDcols = changeCols]
  # str(in.dt)
}
