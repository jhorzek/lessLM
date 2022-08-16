#ifndef LINEARREGRESSIONMODEL_H
#define LINEARREGRESSIONMODEL_H

#include <RcppArmadillo.h>

// [[Rcpp :: depends ( RcppArmadillo )]]

double sumSquaredError(
  arma::colvec b, // the parameter vector
  arma::colvec y, // the dependent variable
  arma::mat X // the design matrix
  );

arma::rowvec sumSquaredErrorGradients(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X // the design matrix
);

// we could also define the analytic Hessian, but we
// are a bit lazy here. The Hessian is only required
// As a starting point for the BFGS approximations. However,
// this starting point can have a huge impact.
arma::mat approximateHessian(arma::colvec b, // the parameter vector
                             arma::colvec y, // the dependent variable
                             arma::mat X, // the design matrix
                             double eps // controls the exactness of the approximation
);
#endif