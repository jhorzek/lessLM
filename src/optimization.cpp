#include <RcppArmadillo.h>
#include "lessSEM.hpp" // It is important that you include the lessSEM.hpp
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
  linearRegressionModel(arma::colvec y_, arma::mat X_):
    y(y_), X(X_){}
  
};

// Step 2: Now that we have defined our model, we can implement the functions
// which are necessary to optimize this model. We will use the elastic net with
// glmnet optimizer as an example.

// [[Rcpp::export]]
Rcpp::NumericVector elasticNet(
    const arma::colvec y, 
    arma::mat X,
    const double alpha,
    const double lambda
)
{
  
  // first, let's add a column to X for our intercept
  X.insert_cols(0,1);
  X.col(0) += 1.0;
  
  // Now we define the parameter vector b
  // This must be an Rcpp::NumericVector with labels
  Rcpp::NumericVector b(X.n_cols,0);
  Rcpp::StringVector bNames;
  // now our vector needs names. This is a bit
  // cumbersome and it would be easier to just let
  // the user pass in a labeled R vector to the elasticNet
  // function. However, we want to make this as convenient
  // as possible for the user. Therefore, we have got to 
  // create these labels:
  for(int i = 0; i < b.length(); i++){
    bNames.push_back("b" + std::to_string(i));
  }
  // add the labels to the parameter vector:
  b.names() = bNames;
  
  // now, it is time to set up the model we defined above
  
  linearRegressionModel linReg(y,X);
  
  // next, we have to define the penalties we want to use.
  // The elastic net is a combination of a ridge penalty and 
  // a lasso penalty:
  lessSEM::penaltyLASSO lasso;
  lessSEM::penaltyRidge ridge;
  // these penalties take tuning parameters of class tuningParametersEnet
  lessSEM::tuningParametersEnet tp;
  tp.alpha = alpha;
  tp.lambda = lambda;
  // finally, there is also the weights. The weights vector indicates, which
  // of the parameters is regularized (weight = 1) and which is unregularized 
  // (weight =0). It also allows for adaptive lasso weights (e.g., weight =.0123).
  // weights must be an arma::rowvec of the same length as our parameter vector.
  arma::rowvec weights(b.length());
  weights.fill(1.0); // we want to regularize all parameters
  weights.at(0) = 0.0; // except for the first one, which is our intercept.
  tp.weights = weights;
  
  // if we want to fine tune the optimizer, we can use the control
  // arguments. We will start with the default control elements and 
  // tweak some arguments to our liking:
  lessSEM::controlGLMNET control = lessSEM::controlGlmnetDefault();
  
  control.breakOuter = 1e-5;
  control.initialHessian = approximateHessian(b,y,X,1e-5);
  
  // to optimize this model, we have to pass it to
  // one of the optimizers in lessSEM. These are 
  // glmnet and ista. We'll use glmnet in the following. The optimizer will
  // return an object of class fitResults which will have the following fields:
  // 1) convergence (boolean)
  // 2) penalizedFit_k -> double with fit values
  // 3) fits: a vector with the fits of all iterations
  // 4) parameterValues the final parameter values as an arma::rowvec
  // 5) Hessian: the BFGS Hessian approximation
  
  lessSEM::fitResults lmFit = lessSEM::glmnet(
    linReg, // the first argument is our model
    b, // the second are the parameters
    lasso, // the third is our lasso penalty
    ridge, // the fourth our ridge penalty
    tp, // the fifth is our tuning parameter 
    control // finally, let's fine tune with the control
  );
  
  if(lmFit.convergence){
    Rcpp::Rcout << "Model converged with final fit value "<< lmFit.fit;
  }
  
  for(int i = 0; i < b.length(); i++){
    b.at(i) = lmFit.parameterValues.at(i);
  }

  return(b);
}


