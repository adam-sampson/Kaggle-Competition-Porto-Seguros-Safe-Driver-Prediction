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
