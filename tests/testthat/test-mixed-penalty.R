test_that("mixed penalty works", {
  set.seed(123)
  library(lessLM)
  
  # let's first define a small print function to beautify our results
  printCoefficients <- function(model){
    print(t(Matrix:::Matrix(model$B, sparse = TRUE)))
  }
  
  # first, we simulate data for our
  # linear regression.
  N <- 100 # number of persons
  p <- 10 # number of predictors
  X <- matrix(rnorm(N*p),	nrow = N, ncol = p) # design matrix
  b <- c(rep(1,4), 
         rep(0,6)) # true regression weights
  y <- X%*%matrix(b,ncol = 1) + rnorm(N,0,.2)
  
  # Our implementation of linear regressions does not automatically add an intercept.
  # Therefore, we will add a column for the intercept:
  Xext <- cbind(1, X)
  
  startingValues <- rep(0, ncol(Xext))
  names(startingValues) <- paste0("b", 0:(ncol(Xext)-1))
  
  lambda_ <- .3
  theta <- 1
  
  mixedGlmnet <- suppressWarnings(lessLM::penalizeGlmnet(
    y = y,
    X = Xext,
    # We pass our starting values:
    startingValues = startingValues,
    # For each of our values, we have to specify the penalty we want to use.
    # Possible are "none", "cappedL1", "lasso", "lsp", "mcp", or "scad":
    penalty = c("none", # we don't want to regularize the intercept
                "cappedL1", "cappedL1", "cappedL1", "cappedL1", "cappedL1", 
                "lasso", "lasso", "lasso", "lasso", "lasso"),
    lambda = lambda_,
    theta = theta, # theta will be used by cappedL1
    initialHessian = matrix(1)
  ))
  
  ff <- function(b,X,y, lambda_, theta_){
    pen <- 0
    for(i in b[2:6]){
      pen <- pen + lambda_*min(abs(i), theta_)
    }
    
    pen <- pen +
      lambda_*sum(abs(b[7:11])) 
    return((1/(2*length(y)))*sum((y-X%*%matrix(b, ncol = 1))^2) + pen)
  }
  
  out <- optim(par = startingValues, 
               fn = ff,
               method = "BFGS", 
               X = Xext, 
               y = y, 
               lambda_ = lambda_, 
               theta_ = theta)
  testthat::expect_equal(all(abs(
    out$par - mixedGlmnet$rawParameters) < 1e-3), TRUE)
  
})
