#include <RcppArmadillo.h>
#include "lessSEM.h" // It is important that you include the lessSEM.h
// file. for this to work, lessSEM must be installed and your DESCRIPTION file
// must have lessSEM in the LinkingTo field.
#include "linearRegressionModel.h" // of course, we also want to include
// our own model

// the first step when linking to lessSEM is to define a model class.
// The procedure is very similar to the ensmallen library, from which
// we have adapted the following approach (see https://ensmallen.org/)
// The model MUST inherit from the model class defined in lessSEM.
// this class lives in the lessSEM namespace and is accessed with lessSEM::model

class linearRegressionModel : public lessSEM::model{
  
public:

  // the lessSEM::model class has two methods: "fit" and "gradients".
  // Both of these methods must follow a fairly strict framework.
  // First: They must receive exactly two arguments: 
  //        1) an arma::rowvec with current parameter values
  //        2) an Rcpp::StringVector with current parameter labels 
  //          (NOTE: the lessSEM package currently does not make use of these labels.
  //                 This is just for future use. If you don't want to use the labels,
  //                just pass any Rcpp::StringVector you want).
  // Second:
  //        1) fit must return a double (e.g., the -2-log-likelihood) 
  //        2) gradients must return an arma::rowvec with the gradients. It is
  //           important that the gradients are returned in the same order as the 
  //           parameters (i.e., don't shuffle your gradients, lessSEM will assume
  //           that the first value in gradients corresponds to the derivative with
  //           respect to the first parameter passed to the function).
  
  double fit(arma::rowvec b, Rcpp::StringVector labels) override{
    //NOTE: In sumSquaredError we assumed that b was a column-vector. We
    // have to transpose b to make things work
    return(sumSquaredError(b.t(), y, X));
  }
  
  arma::rowvec gradients(arma::rowvec b, Rcpp::StringVector labels) override{
    //NOTE: In sumSquaredErrorGradients we assumed that b was a column-vector. We
    // have to transpose b to make things work
    return(sumSquaredErrorGradients(b.t(), y, X));
  }
  
  // IMPORTANT: Note that we used some arguments above which we did not pass to
  // the functions: y, and X. Without these arguments, we cannot use our
  // sumSquaredError and sumSquaredErrorGradients function! To make these accessible
  // to our functions, we have to define them:
  
  const arma::colvec y;
  const arma::mat X;
  
  // finally, we create a constructor for our class
  linearRegressionModel(arma::colvec y_, arma::mat X_): y(y_), X(X_){};
  
};


// Using glmnet
// [[Rcpp::export]]
Rcpp::List penalizeGlmnet(arma::colvec y, 
                    arma::mat X,
                    Rcpp::NumericVector startingValues,
                    std::vector<std::string> penalty,
                    arma::rowvec lambda,
                    arma::rowvec theta,
                    arma::mat initialHessian){
  
  // With that, we can create our model:
  linearRegressionModel linReg(y,X);
  
  // You should also provide an initial Hessian as this can 
  // improve the optimization considerably. For our model, this 
  // Hessian can be computed with the Hessian function defined 
  // in linerRegressionModel.cpp. For simplicity, we will
  // use the default Hessian as an example below. This will be
  // an identity matrix which will work in our simple example but may fail
  // in other cases. In practice, you should consider using a better approach.
  // arma::colvec val = Rcpp::as<arma::colvec>(startingValues);
  // arma::mat initialHessian = approximateHessian(val,
  //                                               y,
  //                                               X,
  //                                               1e-5
  // );
  
  // and optimize:
  lessSEM::fitResults fitResult_ = lessSEM::fitGlmnet(
    linReg,
    startingValues,
    penalty,
    lambda,
    theta,
    initialHessian // optional, but can be very useful
  );
  
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("fit") = fitResult_.fit,
    Rcpp::Named("convergence") = fitResult_.convergence,
    Rcpp::Named("rawParameters") = fitResult_.parameterValues,
    Rcpp::Named("Hessian") = fitResult_.Hessian
  );
  return(result);
  
}

// Setting up optimization with Ista works similarly:

// Using ista
// [[Rcpp::export]]
Rcpp::List penalizeIsta(arma::colvec y, 
                        arma::mat X,
                        Rcpp::NumericVector startingValues,
                        std::vector<std::string> penalty,
                        arma::rowvec lambda,
                        arma::rowvec theta){
  
  // With that, we can create our model:
  linearRegressionModel linReg(y,X);
  
  // and optimize:
  lessSEM::fitResults fitResult_ = lessSEM::fitIsta(
    linReg,
    startingValues,
    penalty,
    lambda,
    theta
  );
  
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("fit") = fitResult_.fit,
    Rcpp::Named("convergence") = fitResult_.convergence,
    Rcpp::Named("rawParameters") = fitResult_.parameterValues
  );
  return(result);
  
}

