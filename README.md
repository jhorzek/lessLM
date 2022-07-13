
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
                 alpha = 1, # note: glmnet and lessSEM define 
                 # the elastic net differently (lessSEM follows lslx and regsem)
                 # Therefore, you will get different results if you change alpha
                 # when compared to glmnet
                 lambda = seq(1,0,-.1))

library(glmnet)
#> Loading required package: Matrix
#> Loaded glmnet 4.1-4
l2 <- glmnet(x = X, 
             y = y, 
             lambda = seq(0,1,.1),
             standardize = FALSE)
coef(l2)
#> 11 x 11 sparse Matrix of class "dgCMatrix"
#>    [[ suppressing 11 column names 's0', 's1', 's2' ... ]]
#>                                                                           
#> (Intercept) 0.09341722 0.1016531 0.1160556 0.12462096 0.1134846 0.09911437
#> V1          .          .         .         .          0.1171166 0.26672659
#> V2          .          0.0456787 0.1494955 0.25178426 0.3567714 0.46309006
#> V3          .          .         .         0.04769799 0.1757818 0.30792600
#> V4          0.06426310 0.1560080 0.2453808 0.33677270 0.4358888 0.53637577
#> V5          .          .         .         .          .         .         
#> V6          .          .         .         .          .         .         
#> V7          .          .         .         .          .         .         
#> V8          .          .         .         .          .         .         
#> V9          .          .         .         .          .         .         
#> V10         .          .         .         .          .         .         
#>                                                                     
#> (Intercept) 0.08474437 0.07037437 0.05600437 0.04163437  0.027387180
#> V1          0.41633553 0.56594447 0.71555342 0.86516236  1.012910134
#> V2          0.56940931 0.67572857 0.78204782 0.88836707  0.999125705
#> V3          0.44006956 0.57221313 0.70435670 0.83650026  0.970569587
#> V4          0.63686264 0.73734950 0.83783637 0.93832324  1.027638033
#> V5          .          .          .          .           0.014021011
#> V6          .          .          .          .          -0.007451938
#> V7          .          .          .          .           0.018592112
#> V8          .          .          .          .           0.021926877
#> V9          .          .          .          .          -0.009900589
#> V10         .          .          .          .           0.027396435
t(l1$B)
#>           [,1]       [,2]      [,3]      [,4]      [,5]       [,6]       [,7]
#> b0  0.09348642 0.10166404 0.1160315 0.1246996 0.1135478 0.09912052 0.08477288
#> b1  0.00000000 0.00000000 0.0000000 0.0000000 0.1171302 0.26665223 0.41633238
#> b2  0.00000000 0.04567918 0.1495003 0.2517965 0.3567501 0.46311060 0.56941823
#> b3  0.00000000 0.00000000 0.0000000 0.0473653 0.1757680 0.30793040 0.44004914
#> b4  0.06423391 0.15602722 0.2453946 0.3368144 0.4359944 0.53639804 0.63686256
#> b5  0.00000000 0.00000000 0.0000000 0.0000000 0.0000000 0.00000000 0.00000000
#> b6  0.00000000 0.00000000 0.0000000 0.0000000 0.0000000 0.00000000 0.00000000
#> b7  0.00000000 0.00000000 0.0000000 0.0000000 0.0000000 0.00000000 0.00000000
#> b8  0.00000000 0.00000000 0.0000000 0.0000000 0.0000000 0.00000000 0.00000000
#> b9  0.00000000 0.00000000 0.0000000 0.0000000 0.0000000 0.00000000 0.00000000
#> b10 0.00000000 0.00000000 0.0000000 0.0000000 0.0000000 0.00000000 0.00000000
#>           [,8]       [,9]      [,10]        [,11]
#> b0  0.07051992 0.05643675 0.04164992  0.027727456
#> b1  0.56594941 0.71574719 0.86512441  1.012718764
#> b2  0.67575754 0.78245673 0.88836766  0.998883257
#> b3  0.57216451 0.70387969 0.83653460  0.970518965
#> b4  0.73736836 0.83813706 0.93837885  1.027613864
#> b5  0.00000000 0.00000000 0.00000000  0.013057572
#> b6  0.00000000 0.00000000 0.00000000 -0.007160196
#> b7  0.00000000 0.00000000 0.00000000  0.018649256
#> b8  0.00000000 0.00000000 0.00000000  0.021975303
#> b9  0.00000000 0.00000000 0.00000000 -0.010585376
#> b10 0.00000000 0.00000000 0.00000000  0.027377045
```
