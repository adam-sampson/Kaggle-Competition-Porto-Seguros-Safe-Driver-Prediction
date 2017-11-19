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
