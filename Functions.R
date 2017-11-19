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
