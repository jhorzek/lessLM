
<!-- README.md is generated from README.Rmd. Please edit that file -->

# lessLM

The sole purpose of this package is to demonstrate how the optimizers
implemented in [lessSEM](https://github.com/jhorzek/lessSEM) can be used
by other packages. To this end, we use a fairly simple model: A linear
regression of the form

![\\pmb y = \\pmb X \\pmb b + \\pmb\\varepsilon](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cpmb%20y%20%3D%20%5Cpmb%20X%20%5Cpmb%20b%20%2B%20%5Cpmb%5Cvarepsilon "\pmb y = \pmb X \pmb b + \pmb\varepsilon")

## Step 1

Install [lessSEM](https://github.com/jhorzek/lessSEM) from
<https://github.com/jhorzek/lessSEM>.

## Step 2

Create a new R package which uses
[RcppArmadillo](https://github.com/RcppCore/RcppArmadillo) (e.g., with
`RcppArmadillo::RcppArmadillo.package.skeleton()`). Open the DESCRIPTION
file and add lessSEM to the “LinkingTo” field (see the DESCRIPTION file
of this package if you are unsure what we are referring to). If you
already have a package, just add lessSEM to the “LinkingTo” field.

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
src/linearRegressionModel.h and src/linearRegressionModel.cpp. We also
implemented a function to compute the Hessian. This function is very
helpful when using the glmnet optimizer which does use an approximation
of the Hessian based on the BFGS procedure. If the initial Hessian is a
poor substitute for the true Hessian, this optimizer can return wrong
parameter estimates. To get a reasonable initial Hessian, we therefore
also implemented this initial Hessian estimation based on the procedure
used in [lavaan](https://github.com/yrosseel/lavaan).

## Step 4: Linking to [lessSEM](https://github.com/jhorzek/lessSEM)

Assuming that our model is set up, we are ready to link everything to
[lessSEM](https://github.com/jhorzek/lessSEM). First, create a new file
(we called ours src/optimization.cpp). Here, it is important to include
the lessSEM.hpp headers (see src/optimization.cpp). All further steps
necessary to use the [lessSEM](https://github.com/jhorzek/lessSEM)
optimizers are outlined in the comments included in the new file
src/optimization.cpp. Open this file and follow the instructions.
Therein, we implement both, glmnet and ista optimization for the elastic
net penalty and ista optimization of the scad penalty. If you are
interested in how the optimization routine is designed, have a look at
the vignette “The-optimizer-interface” of the
[lessSEM](https://github.com/jhorzek/lessSEM) package.

## Step 5: Test your function

### Elastic Net

We will compare our results to those of
[glmnet](https://github.com/cran/glmnet).

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

# define the tuning parameters
lambda = seq(1,0,length.out = 5)

lasso1 <- lessLM::elasticNet(y = y,
                 X = X,
                 alpha = 1, # note: glmnet and lessSEM define 
                 # the elastic net differently (lessSEM follows lslx and regsem)
                 # Therefore, you will get different results if you change alpha
                 # when compared to glmnet
                 lambda = lambda
                 )

# now, let's use the ista optimizer
lasso2 <- lessLM::elasticNetIsta(y = y,
                 X = X,
                 alpha = 1, # note: glmnet and lessSEM define 
                 # the elastic net differently (lessSEM follows lslx and regsem)
                 # Therefore, you will get different results if you change alpha
                 # when compared to glmnet
                 lambda = lambda)

# For comparison, we will fit the model with the glmnet package:
library(glmnet)
#> Loaded glmnet 4.1-4
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
#> b0  0.09341722 0.1232568 0.09911445 0.0631894  0.027385203
#> b1  .          .         0.26672653 0.6407490  1.012920701
#> b2  .          0.2014038 0.46309007 0.7288881  0.999144918
#> b3  .          .         0.30792603 0.6382850  0.970572420
#> b4  0.06426309 0.2900671 0.53637576 0.7875929  1.027626352
#> b5  .          .         .          .          0.014036323
#> b6  .          .         .          .         -0.007460361
#> b7  .          .         .          .          0.018590428
#> b8  .          .         .          .          0.021930166
#> b9  .          .         .          .         -0.009899935
#> b10 .          .         .          .          0.027400904
printCoefficients(lasso2)
#> 11 x 5 sparse Matrix of class "dgCMatrix"
#>                                                            
#> b0  0.09341660 0.1232553 0.09911708 0.06319168  0.027380698
#> b1  .          .         0.26672763 0.64074985  1.012922529
#> b2  .          0.2014050 0.46308822 0.72888662  0.999147143
#> b3  .          .         0.30792723 0.63828598  0.970570230
#> b4  0.06426509 0.2900689 0.53637334 0.78759086  1.027637648
#> b5  .          .         .          .           0.014031732
#> b6  .          .         .          .          -0.007464030
#> b7  .          .         .          .           0.018593755
#> b8  .          .         .          .           0.021935357
#> b9  .          .         .          .          -0.009908269
#> b10 .          .         .          .           0.027408142
```

### Scad

Our functions implementing the scad penalty can be found in
src/optimization.cpp. We will compare our function to that of
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

# Now, let's fit our model with the standardized data
scad1 <- lessLM::scadIsta(y = y, 
                         X = X, 
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
#> b0  0.09108804 0.09108943 0.09108943 0.09108943  0.091089427
#> b1  .          .          0.29982150 0.92165332  0.919968554
#> b2  .          0.22333858 0.46569513 0.95702869  0.961318003
#> b3  .          0.03305713 0.32816325 0.91548510  0.917309551
#> b4  0.03393651 0.27562871 0.58291071 1.07368802  1.062121218
#> b5  .          .          .          .           0.013824173
#> b6  .          .          .          .          -0.006963821
#> b7  .          .          .          .           0.019019983
#> b8  .          .          .          .           0.022031383
#> b9  .          .          .          .          -0.010358121
#> b10 .          .          .          .           0.027812833
```

# References

-   Rosseel, Y. (2012). lavaan: An R Package for Structural Equation
    Modeling. Journal of Statistical Software, 48(2), 1–36.
    <https://doi.org/10.18637/jss.v048.i02>
-   Breheny, P. (2021). ncvreg: Regularization paths for scad and mcp
    penalized regression models.
-   Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization
    paths for generalized linear models via coordinate descent. Journal
    of Statistical Software, 33(1), 1–20.
    <https://doi.org/10.18637/jss.v033.i01>
