##########
# File to load functions cleanly outside main program(s)
##########

loadPackages <- function(packsToLoad) {
  for(package in packsToLoad) {
    if(require(package, character.only = TRUE) == FALSE) {
      install.packages(package)
      Sys.sleep(3)
      require(package, character.only = TRUE)
    } else {
      require(package, character.only = TRUE)
    }
  }
}

# Note: use gc() to clean up memory...
# Note: use gc(verbose=TRUE) to clean up more
# Note: use memory.size() to see how much memory is being used...

# checkForValuesInBoth <- function(sourceList, checkList) {
#   outList <- list()
#   for(feature in names(sourceList)) {
#     # print(paste("Feature:",feature))
#     tempCatch <- NULL
#     for(value in sourceList[[feature]]) {
#       # print(paste("Value:",value))
#       if(!(value %in% checkList[[feature]])) {
#         tempCatch <- c(tempCatch,value)
#       } #else {
#         # Do nothing.
#         # warning(paste0("Match found for ", value, " in ", feature))
#       # }
#     }
#     # print(paste("TempCatch = ",c(tempCatch)))
#     # if( (is.null(tempCatch) == FALSE) & (exists("outList", inherits = FALSE)) ) {
#     #   outList <- c(outList,list(eval(feature) = tempCatch))
#     # } else if (is.null(tempCatch) == FALSE) {
#     #   outList <- list(eval(feature) = tempCatch)
#     # }
#     outList[[feature]] <- tempCatch
#   }
#   # if(exists("outList", inherits = FALSE)){
#   #   return(outList)
#   # }
#   # return(NULL)
#   return(outList)
# }

checkForValuesInBoth <- function(sourceList, checkList) {
  outList <- list()
  for(feature in names(sourceList)) {
    # print(paste("Feature:",feature))
    tempCatch <- NULL
    logical.check <- sourceList[[feature]] %in% checkList[[feature]]
    tempCatch <- sourceList[[feature]][!logical.check]
    # if(sum(as.numeric(tempCatch)) < length(tempCatch)) {  
      outList[[feature]] <- tempCatch
    # }
  }
  return(outList)
}

# Let's try something different. What happens with
# unique.test[[22]] %in% unique.train[[23]]
tempvar <- unique.test[["ps_reg_03"]] %in% unique.train[["ps_reg_03"]]
unique.test[["ps_reg_03"]][!tempvar]

tempvar <- unique.test[["ps_reg_02"]] %in% unique.train[["ps_reg_02"]]
unique.test[["ps_reg_02"]][!tempvar]
