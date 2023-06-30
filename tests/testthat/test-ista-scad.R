test_that("testing ista scad", {
  
  set.seed(123)
  library(lessLM)
  library(ncvreg)
  
  # first, we simulate data for our
  # linear regression.
  N <- 100 # number of persons
  p <- 10 # number of predictors
  X <- matrix(rnorm(N*p),	nrow = N, ncol = p) # design matrix
  b <- c(rep(1,4), 
         rep(0,6)) # true regression weights
  y <- X%*%matrix(b,ncol = 1) + rnorm(N,0,.2)
  
  X <- ncvreg::std(X)
  attr(X, "center") <- NULL
  attr(X, "scale") <- NULL
  attr(X, "nonsingular") <- NULL
  
  Xext <- cbind(1, X)
  
  startingValues <- rep(0, ncol(Xext))
  names(startingValues) <- paste0("b", 0:(ncol(Xext)-1))
  
  # define the tuning parameters
  lambda = seq(1,0,length.out = 5)
  
  # Now, let's fit our model with the standardized data
  scad1 <- lessLM::scadIsta(y = y, 
                            X = Xext, 
                            startingValues = startingValues,
                            theta = 3, 
                            lambda = lambda)
  
  # iterate over multiple lambdas
  initialHessian <- matrix(1)
  estimates <- c()
  for(l in lambda){
    fit <- suppressWarnings(lessLM::penalizeGlmnet(
      y = y,
      X = Xext,
      startingValues = startingValues,
      penalty = c("none", 
                  "scad", "scad", "scad", "scad", "scad", 
                  "scad", "scad", "scad", "scad", "scad"),
      lambda = l, 
      theta = 3,
      initialHessian = initialHessian
    )
    )
    # save estimates
    estimates <- rbind(estimates, fit$rawParameters)
    # update Hessian for next iterations
    initialHessian = fit$Hessian
  }
  
  # for comparison, we use ncvreg
  scadFit <- ncvreg(X = X, 
                    y = y, 
                    penalty = "SCAD",
                    lambda = lambda, 
                    gamma = 3)
  
  testthat::expect_equal(all(abs(coef(scadFit) -
                                   t(scad1$B)) < 1e-4), TRUE)
  testthat::expect_equal(all(abs(coef(scadFit) -
                                   t(estimates)) < 1e-4), TRUE)
  
})