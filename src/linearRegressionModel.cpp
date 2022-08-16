#include <RcppArmadillo.h>
#include "linearRegressionModel.h"

// [[Rcpp :: depends ( RcppArmadillo )]]

// [[Rcpp::export]]
double sumSquaredError(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X // the design matrix
){
  // compute the sum of squared errors:
  arma::mat sse = arma::trans(y-X*b)*(y-X*b);
  
  // other packages, such as glmnet, scale the sse with 
  // 1/(2*N), where N is the sample size. We will do that here as well
  
  sse *= 1.0/(2.0 * y.n_elem);
  
  // note: We must return a double, but the sse is a matrix
  // To get a double, just return the single value that is in 
  // this matrix:
  return(sse(0,0));
}

// [[Rcpp::export]]
arma::rowvec sumSquaredErrorGradients(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X // the design matrix
){
  // note: we want to return our gradients as row-vector; therefore,
  // we have to transpose the resulting column-vector:
  arma::rowvec gradients = arma::trans(-2.0*X.t() * y + 2.0*X.t()*X*b);
  
  // other packages, such as glmnet, scale the sse with 
  // 1/(2*N), where N is the sample size. We will do that here as well
  
  gradients *= (.5/y.n_rows);
  
  return(gradients);
}

// [[Rcpp::export]]
arma::mat approximateHessian(arma::colvec b, // the parameter vector
                             arma::colvec y, // the dependent variable
                             arma::mat X, // the design matrix
                             double eps // controls the exactness of the approximation
                               ){
  int nPar = b.n_elem;
  arma::mat hessian(nPar, nPar, arma::fill::zeros);
  
  arma::colvec stepLeft = b, 
    twoStepLeft = b, 
    stepRight = b, 
    twoStepRight = b;
  
  arma::rowvec gradientsStepLeft(nPar);
  arma::rowvec gradientsTwoStepLeft(nPar);
  arma::rowvec gradientsStepRight(nPar);
  arma::rowvec gradientsTwoStepRight(nPar);
  
  // THE FOLLOWING CODE IS ADAPTED FROM LAVAAN. 
  // SEE lavaan:::lav_model_hessian FOR THE IMPLEMENTATION
  // BY Yves Rosseel
  
  for(int p = 0; p < nPar; p++) {
    
    stepLeft.at(p) -= eps;
    twoStepLeft.at(p) -= 2*eps;
    stepRight.at(p) += eps;
    twoStepRight.at(p) += 2*eps;
    
    // step left
    gradientsStepLeft = sumSquaredErrorGradients(stepLeft, y, X);
    
    // two step left
    gradientsTwoStepLeft = sumSquaredErrorGradients(twoStepLeft, y, X);
    
    // step right
    gradientsStepRight = sumSquaredErrorGradients(stepRight, y, X);
    
    // two step right
    gradientsTwoStepRight = sumSquaredErrorGradients(twoStepRight, y, X);
    
    // approximate hessian
    hessian.col(p) = arma::trans((gradientsTwoStepLeft - 
      8.0 * gradientsStepLeft + 
      8.0 * gradientsStepRight - 
      gradientsTwoStepRight)/(12.0 * eps));
    
    // reset
    stepLeft.at(p) += eps;
    twoStepLeft.at(p) += 2*eps;
    stepRight.at(p) -= eps;
    twoStepRight.at(p) -= 2*eps;
  }
  // make symmetric
  hessian = (hessian + arma::trans(hessian))/2.0;

  return(hessian);
}
