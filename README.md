
<!-- README.md is generated from README.Rmd. Please edit that file -->

# lessLM

The sole purpose of this package is to demonstrate how the optimizers
implemented in [lessSEM](https://github.com/jhorzek/lessSEM) can be used
by other packages. To this end, we use a fairly simple model: A linear
regression of the form

$$\pmb y = \pmb X \pmb b + \pmb\varepsilon$$

## Step 1

Install [lessSEM](https://github.com/jhorzek/lessSEM) from CRAN or
[GitHub](https://github.com/jhorzek/lessSEM).

``` r
install.packages("lessSEM")
```

## Step 2

Create a new R package which uses
[RcppArmadillo](https://github.com/RcppCore/RcppArmadillo) (e.g., with
`RcppArmadillo::RcppArmadillo.package.skeleton()`). Open the DESCRIPTION
file and add lessSEM to the “LinkingTo” field (see the DESCRIPTION file
of this package if you are unsure what we are referring to). If you
already have a package, just add lessSEM to the “LinkingTo” field.

Alternatively, you can copy the optimizers from
[here](https://github.com/jhorzek/lessOptimizers) into the source folder
of your package.

## Step 3: Implementing your package

We will assume that you already have a package which implements the
model that you would like to regularize. Therefore, we will not make use
of any of the functions and classes defined in
[lessSEM](https://github.com/jhorzek/lessSEM) at this point. However,
make sure that you have two functions:

1)  A function which computes the fit (e.g., -2-log-Likelihood) of your
    model. This function must return a double.
2)  A function which computes the gradients of your model. This function
    must return an arma::rowvec.

For our example, we have implemented these functions in the files
[src/linearRegressionModel.h](https://github.com/jhorzek/lessLM/blob/main/src/linearRegressionModel.h)
and
[src/linearRegressionModel.cpp](https://github.com/jhorzek/lessLM/blob/main/src/linearRegressionModel.cpp).
We also implemented a function to compute the Hessian. This function is
very helpful when using the glmnet optimizer which does use an
approximation of the Hessian based on the BFGS procedure. If the initial
Hessian is a poor substitute for the true Hessian, this optimizer can
return wrong parameter estimates. To get a reasonable initial Hessian,
we therefore also implemented this initial Hessian estimation based on
the procedure used in [lavaan](https://github.com/yrosseel/lavaan).

## Step 4: Linking to [lessSEM](https://github.com/jhorzek/lessSEM)

Assuming that our model is set up, we are ready to link everything to
[lessSEM](https://github.com/jhorzek/lessSEM). First, create a new file
(we called ours
[src/optimization.cpp](https://github.com/jhorzek/lessLM/blob/main/src/optimization.cpp)).
Here, it is important to include the lessSEM.h headers (see
[src/optimization.cpp](https://github.com/jhorzek/lessLM/blob/main/src/optimization.cpp)).
All further steps necessary to use the
[lessSEM](https://github.com/jhorzek/lessSEM) optimizers are outlined in
the comments included in the new file
[src/optimization.cpp](https://github.com/jhorzek/lessLM/blob/main/src/optimization.cpp).
Open this file and follow the instructions. Therein, we implement both,
glmnet and ista optimization for mixed penalties. You can also implement
more specialized routines. This is shown in the file
[src/optimization_specialized.cpp](https://github.com/jhorzek/lessLM/blob/main/src/optimization_specialized.cpp),
where the elastic net penalty and ista optimization of the scad penalty
are demonstrated. If you are interested in how the optimization routine
is designed, have a look at the vignette “The-optimizer-interface” of
the [lessSEM](https://github.com/jhorzek/lessSEM) package.

## Step 5: Test your function

### Simulating data

``` r
set.seed(123)
library(lessLM)
library(Matrix) # we will use the matrix package
# to print the matrices below in the same sparse format
# as glmnet or ncvreg

# let's first define a small print function to beautify our results
printCoefficients <- function(model){
  print(t(Matrix:::Matrix(model$B, sparse = TRUE)))
}

# first, we simulate data for our
# linear regression.
N <- 100 # number of persons
p <- 10 # number of predictors
X <- matrix(rnorm(N*p), nrow = N, ncol = p) # design matrix
b <- c(rep(1,4), 
       rep(0,6)) # true regression weights
y <- X%*%matrix(b,ncol = 1) + rnorm(N,0,.2)

# Our implementation of linear regressions does not automatically add an intercept.
# Therefore, we will add a column for the intercept:
Xext <- cbind(1, X)
```

### Using the functions in [src/optimization.cpp](https://github.com/jhorzek/lessLM/blob/main/src/optimization.cpp)

The implementation in
[src/optimization.cpp](https://github.com/jhorzek/lessLM/blob/main/src/optimization.cpp)
is very flexible as it allows for combining different penalties for
different parameters. To use the function, we have to first provide some
starting values:

``` r
startingValues <- rep(0, ncol(Xext))
names(startingValues) <- paste0("b", 0:(ncol(Xext)-1))
```

Now we can regularize our model:

``` r
# glmnet
lassoGlmnet <- lessLM::penalizeGlmnet(
  y = y,
  X = Xext,
  # We pass our starting values:
  startingValues = startingValues,
  # For each of our values, we have to specify the penalty we want to use.
  # Possible are "none", "cappedL1", "lasso", "lsp", "mcp", or "scad":
  penalty = c("none", # we don't want to regularize the intercept
              # all other 10 regression parameters are regularized with the lasso:
              "lasso", "lasso", "lasso", "lasso", "lasso", 
              "lasso", "lasso", "lasso", "lasso", "lasso"),
  lambda = .3, # the same lambda will be used for all parameters. We can also pass
  # parameter-specific lambda values
  theta = 0, # Theta is not used by any of the penalties, but we still have to pass
  # some values
  initialHessian = matrix(1)
)
#> Warning in lessLM::penalizeGlmnet(y = y, X = Xext, startingValues =
#> startingValues, : Setting initial Hessian to identity matrix. We recommend
#> passing a better Hessian.
```

Note that we get a warning because we did not pass a true Hessian. For
our simple model, this will still work.

We can check the paramters as follows:

``` r
Matrix::Matrix(lassoGlmnet$rawParameters,
               sparse = TRUE)
#> 1 x 11 sparse Matrix of class "dgCMatrix"
#>                                                                  
#> [1,] 0.070374 0.5659441 0.6757283 0.5722129 0.7373493 . . . . . .
```

In practice, you will want to use multiple $\lambda$ values:

``` r
# iterate over multiple lambdas
initialHessian <- matrix(1)
estimates <- c()
for(lambda in seq(0,1,.1)){
  fit <- lessLM::penalizeGlmnet(
    y = y,
    X = Xext,
    startingValues = startingValues,
    penalty = c("none", 
                "lasso", "lasso", "lasso", "lasso", "lasso", 
                "lasso", "lasso", "lasso", "lasso", "lasso"),
    lambda = lambda, 
    theta = 0,
    initialHessian = initialHessian
  )
  # save estimates
  estimates <- rbind(estimates, fit$rawParameters)
  # update Hessian for next iterations
  initialHessian = fit$Hessian
}
round(Matrix:::Matrix(estimates, sparse = TRUE),3)
#> 11 x 11 sparse Matrix of class "dgCMatrix"
#>                                                                         
#>  [1,] 0.027 1.013 0.999 0.971 1.028 0.014 -0.007 0.019 0.022 -0.01 0.027
#>  [2,] 0.042 0.865 0.888 0.836 0.938 .      .     .     .      .    .    
#>  [3,] 0.056 0.716 0.782 0.704 0.838 .      .     .     .      .    .    
#>  [4,] 0.070 0.566 0.676 0.572 0.737 .      .     .     .      .    .    
#>  [5,] 0.085 0.416 0.569 0.440 0.637 .      .     .     .      .    .    
#>  [6,] 0.099 0.267 0.463 0.308 0.536 .      .     .     .      .    .    
#>  [7,] 0.113 0.117 0.357 0.176 0.436 .      .     .     .      .    .    
#>  [8,] 0.125 .     0.252 0.048 0.337 .      .     .     .      .    .    
#>  [9,] 0.116 .     0.149 .     0.245 .      .     .     .      .    .    
#> [10,] 0.102 .     0.046 .     0.156 .      .     .     .      .    .    
#> [11,] 0.093 .     .     .     0.064 .      .     .     .      .    .
```

We can also mix penalties…

``` r
mixedGlmnet <- lessLM::penalizeGlmnet(
  y = y,
  X = Xext,
  # We pass our starting values:
  startingValues = startingValues,
  # For each of our values, we have to specify the penalty we want to use.
  # Possible are "none", "cappedL1", "lasso", "lsp", "mcp", or "scad":
  penalty = c("none", # we don't want to regularize the intercept
              "cappedL1", "cappedL1", "cappedL1", "cappedL1", "cappedL1", 
              "lasso", "lasso", "lasso", "lasso", "lasso"),
  lambda = .3,
  theta = 1, # theta will be used by cappedL1
  initialHessian = matrix(1)
)

round(Matrix:::Matrix(mixedGlmnet$rawParameters, sparse = TRUE),3)
#> 1 x 11 sparse Matrix of class "dgCMatrix"
#>                                              
#> [1,] 0.07 0.566 0.676 0.572 0.737 . . . . . .
```

… or use the ista optimizer instead:

``` r
mixedIsta <- lessLM::penalizeIsta(
  y = y,
  X = Xext,
  # We pass our starting values:
  startingValues = startingValues,
  # For each of our values, we have to specify the penalty we want to use.
  # Possible are "none", "cappedL1", "lasso", "lsp", "mcp", or "scad":
  penalty = c("none", # we don't want to regularize the intercept
              "cappedL1", "cappedL1", "cappedL1", "cappedL1", "cappedL1", 
              "lasso", "lasso", "lasso", "lasso", "lasso"),
  lambda = .3,
  theta = 1 # theta will be used by cappedL1
)

round(Matrix:::Matrix(mixedIsta$rawParameters, sparse = TRUE),3)
#> 1 x 11 sparse Matrix of class "dgCMatrix"
#>                                              
#> [1,] 0.07 0.566 0.676 0.572 0.737 . . . . . .
```

Finally, if we just specify a single penalty, the same penalty will be
used for all parameters:

``` r
mixedIsta <- lessLM::penalizeIsta(
  y = y,
  X = Xext,
  # We pass our starting values:
  startingValues = startingValues,
  penalty = "lasso", # lasso will be applied to all parameters (including 
  # the intercept)
  lambda = .3,
  theta = 0
)

round(Matrix:::Matrix(mixedIsta$rawParameters, sparse = TRUE),3)
#> 1 x 11 sparse Matrix of class "dgCMatrix"
#>                                           
#> [1,] . 0.574 0.668 0.583 0.736 . . . . . .
```

### Specialized Implementations in [src/optimization_specialized.cpp](https://github.com/jhorzek/lessLM/blob/main/src/optimization_specialized.cpp)

Our specialized implementations fit the model for different values of
the tuning parameters.

#### Elastic Net

We will compare our results to those of
[glmnet](https://github.com/cran/glmnet).

``` r
# define the tuning parameters
lambda = seq(1,0,length.out = 5)

lasso1 <- lessLM::elasticNet(y = y,
                             X = Xext,
                             startingValues = startingValues,
                             alpha = 1, # note: glmnet and lessSEM define 
                             # the elastic net differently (lessSEM follows lslx and regsem)
                             # Therefore, you will get different results if you change alpha
                             # when compared to glmnet
                             lambda = lambda
)

# now, let's use the ista optimizer
lasso2 <- lessLM::elasticNetIsta(y = y,
                                 X = Xext,
                                 startingValues = startingValues,
                                 alpha = 1, # note: glmnet and lessSEM define 
                                 # the elastic net differently (lessSEM follows lslx and regsem)
                                 # Therefore, you will get different results if you change alpha
                                 # when compared to glmnet
                                 lambda = lambda)

# For comparison, we will fit the model with the glmnet package:
library(glmnet)
#> Loaded glmnet 4.1-7
lassoGlmnet <- glmnet(x = X, 
                      y = y, 
                      lambda = lambda,
                      standardize = FALSE)
coef(lassoGlmnet)
#> 11 x 5 sparse Matrix of class "dgCMatrix"
#>                     s0        s1         s2         s3           s4
#> (Intercept) 0.09341722 0.1232568 0.09911434 0.06318934  0.027387116
#> V1          .          .         0.26672666 0.64074904  1.012908797
#> V2          .          0.2014030 0.46308997 0.72888809  0.999126195
#> V3          .          .         0.30792610 0.63828503  0.970569484
#> V4          0.06426310 0.2900672 0.53637578 0.78759295  1.027636504
#> V5          .          .         .          .           0.014021661
#> V6          .          .         .          .          -0.007452699
#> V7          .          .         .          .           0.018591741
#> V8          .          .         .          .           0.021927486
#> V9          .          .         .          .          -0.009900790
#> V10         .          .         .          .           0.027396836
printCoefficients(lasso1)
#> 11 x 5 sparse Matrix of class "dgCMatrix"
#>                                                              
#>  [1,] 0.09341722 0.1232569 0.09911444 0.06318939  0.027385625
#>  [2,] .          .         0.26672660 0.64074886  1.012920355
#>  [3,] .          0.2014038 0.46309002 0.72888814  0.999144649
#>  [4,] .          .         0.30792603 0.63828480  0.970572449
#>  [5,] 0.06426310 0.2900671 0.53637578 0.78759293  1.027626673
#>  [6,] .          .         .          .           0.014034808
#>  [7,] .          .         .          .          -0.007459845
#>  [8,] .          .         .          .           0.018590388
#>  [9,] .          .         .          .           0.021929582
#> [10,] .          .         .          .          -0.009900567
#> [11,] .          .         .          .           0.027400801
printCoefficients(lasso2)
#> 11 x 5 sparse Matrix of class "dgCMatrix"
#>                                                              
#>  [1,] 0.09341617 0.1232553 0.09911708 0.06319168  0.027380698
#>  [2,] .          .         0.26672763 0.64074985  1.012922529
#>  [3,] .          0.2014050 0.46308822 0.72888662  0.999147143
#>  [4,] .          .         0.30792723 0.63828598  0.970570230
#>  [5,] 0.06426599 0.2900689 0.53637334 0.78759086  1.027637648
#>  [6,] .          .         .          .           0.014031732
#>  [7,] .          .         .          .          -0.007464030
#>  [8,] .          .         .          .           0.018593755
#>  [9,] .          .         .          .           0.021935357
#> [10,] .          .         .          .          -0.009908269
#> [11,] .          .         .          .           0.027408142
```

#### Scad

Our functions implementing the scad penalty can be found in
[src/optimization_specialized.cpp](https://github.com/jhorzek/lessLM/blob/main/src/optimization_specialized.cpp).
We will compare our function to that of
[ncvreg](https://github.com/pbreheny/ncvreg). Importantly,
[ncvreg](https://github.com/pbreheny/ncvreg) standardizes the data
internally. To use exactly the same data set with both packages, we
apply this standardization first.

``` r
library(ncvreg)
X <- ncvreg::std(X)
attr(X, "center") <- NULL
attr(X, "scale") <- NULL
attr(X, "nonsingular") <- NULL

Xext <- cbind(1, X)

# Now, let's fit our model with the standardized data
scad1 <- lessLM::scadIsta(y = y, 
                          X = Xext,
                          startingValues = startingValues,
                          theta = 3, 
                          lambda = lambda)

# for comparison, we use ncvreg
scadFit <- ncvreg(X = X, 
                  y = y, 
                  penalty = "SCAD",
                  lambda = lambda, 
                  gamma = 3)

coef(scadFit)
#>                 1.0000     0.7500     0.5000     0.2500       0.0000
#> (Intercept) 0.09108943 0.09108943 0.09108943 0.09108943  0.091089427
#> V1          0.00000000 0.00000000 0.29981954 0.92165519  0.919965025
#> V2          0.00000000 0.22333999 0.46569726 0.95702779  0.961297111
#> V3          0.00000000 0.03305662 0.32816203 0.91548202  0.917302145
#> V4          0.03393703 0.27563054 0.58292645 1.07368730  1.062139591
#> V5          0.00000000 0.00000000 0.00000000 0.00000000  0.013801675
#> V6          0.00000000 0.00000000 0.00000000 0.00000000 -0.006960286
#> V7          0.00000000 0.00000000 0.00000000 0.00000000  0.019021092
#> V8          0.00000000 0.00000000 0.00000000 0.00000000  0.022035477
#> V9          0.00000000 0.00000000 0.00000000 0.00000000 -0.010361539
#> V10         0.00000000 0.00000000 0.00000000 0.00000000  0.027813361
printCoefficients(scad1)
#> 11 x 5 sparse Matrix of class "dgCMatrix"
#>                                                               
#>  [1,] 0.09108724 0.09108943 0.09108943 0.09108943  0.091089427
#>  [2,] .          .          0.29982150 0.92165332  0.919968554
#>  [3,] .          0.22333858 0.46569513 0.95702869  0.961318003
#>  [4,] .          0.03305713 0.32816325 0.91548510  0.917309551
#>  [5,] 0.03393716 0.27562871 0.58291071 1.07368802  1.062121218
#>  [6,] .          .          .          .           0.013824173
#>  [7,] .          .          .          .          -0.006963821
#>  [8,] .          .          .          .           0.019019983
#>  [9,] .          .          .          .           0.022031383
#> [10,] .          .          .          .          -0.010358121
#> [11,] .          .          .          .           0.027812833
```

# References

- Rosseel, Y. (2012). lavaan: An R Package for Structural Equation
  Modeling. Journal of Statistical Software, 48(2), 1–36.
  <https://doi.org/10.18637/jss.v048.i02>
- Breheny, P. (2021). ncvreg: Regularization paths for scad and mcp
  penalized regression models.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization
  paths for generalized linear models via coordinate descent. Journal of
  Statistical Software, 33(1), 1–20.
  <https://doi.org/10.18637/jss.v033.i01>
