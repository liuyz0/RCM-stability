gsimp <- matrix(c(
  #0.1915, 0.4172, 0.2255, 0.1658
  #0.0539, 0.5042, 0.3614, 0.0805
  #0.5222, 0.1287, 0.2867, 0.0625,
  0.3456, 0.1947, 0.2362, 0.2235
), byrow = TRUE, nrow = 1, ncol = 4)

csimp <- matrix(c(
  #0.2151, 0.4427, 0.2759, 0.0662
  #0.2239, 0.4088, 0.2756, 0.0916
  #0.3004, 0.2482, 0.3299, 0.1216,
  0.4290, 0.2240, 0.2431, 0.1039
), byrow = TRUE, nrow = 1, ncol = 4)

library(klaR)
#> Loading required package: MASS

quadplot(gsimp)

quadplot(csimp)