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