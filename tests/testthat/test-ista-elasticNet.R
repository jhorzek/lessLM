test_that("testing ista elastic net", {
  
  set.seed(123)
  library(lessLM)
  library(glmnet)
  
  # first, we simulate data for our
  # linear regression.
  N <- 100 # number of persons
  p <- 10 # number of predictors
  X <- matrix(rnorm(N*p),	nrow = N, ncol = p) # design matrix
  b <- c(rep(1,4), 
         rep(0,6)) # true regression weights
  y <- X%*%matrix(b,ncol = 1) + rnorm(N,0,.2)
  
  # define the tuning parameters
  lambda = seq(1,0,length.out = 5)
  
  lasso1 <- lessLM::elasticNetIsta(y = y,
                                   X = X,
                                   alpha = 1, # note: glmnet and lessSEM define 
                                   # the elastic net differently (lessSEM follows lslx and regsem)
                                   # Therefore, you will get different results if you change alpha
                                   # when compared to glmnet
                                   lambda = lambda
  )
  
  # For comparison, we will fit the model with the glmnet package:
  lassoGlmnet <- glmnet(x = X, 
                        y = y, 
                        lambda = lambda,
                        standardize = FALSE)
  
  testthat::expect_equal(all(abs(coef(lassoGlmnet) -
                                   t(lasso1$B)) < 1e-4), TRUE)
  
})