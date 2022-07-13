
<!-- README.md is generated from README.Rmd. Please edit that file -->

# lessLM

The sole purpose of this package is to demonstrate how the optimizers
implemented in lessSEM can be used by other packages. To this end, we
use a fairly simple model: A linear regression of the form

![\\pmb y = \\pmb X \\pmb b + \\pmb\\varepsilon](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cpmb%20y%20%3D%20%5Cpmb%20X%20%5Cpmb%20b%20%2B%20%5Cpmb%5Cvarepsilon "\pmb y = \pmb X \pmb b + \pmb\varepsilon")

## Step 1

Install lessSEM from <https://github.com/jhorzek/lessSEM>.

## Step 2

Create a new R package which uses RcppArmadillo (e.g., with
\`\`RcppArmadillo::RcppArmadillo.package.skeleton()’’). Open the
DESCRIPTION file and add lessSEM to the “LinkingTo” field (see the
DESCRIPTION file of this package if you are unsure what we are referring
to). If you already have a package, just add lessSEM to the “LinkingTo”
field.

## Step 3: Implementing your package

We will assume that you already have a package which implements the
model that you would like to regularize. Therefore, we will not make use
of any of the functions and classes defined in lessSEM at this point.
However, make sure that you have two functions:

1)  A function which computes the fit (e.g., -2-log-Likelihood) of your
    model. This function must return a double.
2)  A function which computes the gradients of your model. This function
    must return an arma::rowvec.

For our example, we have implemented these functions in the files
src/linearRegressionModel.h and src/linearRegressionModel.cpp. We also
implemented a function to compute the Hessian. This function is very
helpful when using the glmnet optimizer which does use an approximation
of the Hessian based on the BFGS procedure. If the initial Hessian is a
poor substitute for the true Hessian, this optimizer can return wrong
parameter estimates. To get a reasonable initial Hessian, we therefore
also implemented this initial Hessian estimation based on the procedure
used in lavaan.

## Step 4: Linking to lessSEM

Assuming that our model is set up, we are ready to link everything to
lessSEM. First, create an new file (we called ours
src/optimization.cpp). Here, it is important to include the lessSEM.hpp
headers (see src/optimization.cpp). All further steps necessary to use
the lessSEM optimizers are outlined in the comments included in the new
file src/optimization.cpp. Open this file and follow the instructions.

## Step 5: Test your function

``` r
set.seed(123)
library(lessLM)

# first, we simulate data for our
# linear regression.
N <- 100 # number of persons
p <- 10 # number of predictors
X <- matrix(rnorm(N*p), nrow = N, ncol = p) # design matrix
b <- c(rep(1,4), 
       rep(0,6)) # true regression weights
y <- X%*%matrix(b,ncol = 1) + rnorm(N,0,.2)

l1 <- elasticNet(y = y,
                  X = X,
                  alpha = 1, 
                  lambda = 1)
#> Model converged with final fit value 1.72747
l1
#>         b0         b1         b2         b3         b4         b5         b6 
#> 0.09341722 0.00000000 0.00000000 0.00000000 0.06426310 0.00000000 0.00000000 
#>         b7         b8         b9        b10 
#> 0.00000000 0.00000000 0.00000000 0.00000000

library(glmnet)
#> Loading required package: Matrix
#> Loaded glmnet 4.1-4
l2 <- glmnet(x = X, 
             y = y, 
             lambda = 1,
             standardize = FALSE)
coef(l2)[,1]
#> (Intercept)          V1          V2          V3          V4          V5 
#>  0.09341722  0.00000000  0.00000000  0.00000000  0.06426310  0.00000000 
#>          V6          V7          V8          V9         V10 
#>  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000
```
