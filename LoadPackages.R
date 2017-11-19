loadPackages <- function(packsToLoad) {
  for(package in packsToLoad) {
    if(require(package, character.only = TRUE) == FALSE) {
      install.packages(package,dependencies = TRUE)
      Sys.sleep(5)
      require(package, character.only = TRUE)
    } else {
      require(package, character.only = TRUE)
    }
  }
}
