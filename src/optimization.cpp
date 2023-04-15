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
  
  // finally, we create a constructor for our class
  linearRegressionModel(arma::colvec y_, arma::mat X_): X(X_), y(y_){};
  
  // IMPORTANT: Note that we used some arguments above which we did not pass to
  // the functions: y, and X. Without these arguments, we cannot use our
  // sumSquaredError and sumSquaredErrorGradients function! To make these accessible
  // to our functions, we have to define them:
  
  const arma::colvec y;
  const arma::mat X;
};

// Step 2: Now that we have defined our model, we can implement the functions
// which are necessary to optimize this model. lessSEM provides two different
// optimizers: glmnet and ista. For each of these optimizers, different penalty 
// functions are implemented. At the time of writing, these are: elastic net, cappedL1,
// lasso, lsp, mcp, and scad. In the first example, we will make use of a simplified
// interface that allows you to hit the ground running. This interface may not be 
// as flexible or fast as you want it to be in some cases, so further below we will
// also demonstrate the use of more specialized interfaces. 

// Using glmnet
// [[Rcpp::export]]
Rcpp::List penalize(arma::colvec y, 
                    arma::mat X,
                    Rcpp::NumericVector startingValues,
                    std::vector<std::string> penalty,
                    arma::rowvec lambda,
                    arma::rowvec theta){
  
  // With that, we can create our model:
  linearRegressionModel linReg(y,X);
  
  // we will also compute an initial Hessian because this can 
  // improve the optimization considerably
  arma::colvec val = Rcpp::as<arma::colvec>(startingValues);
  arma::mat initialHessian = approximateHessian(val,
                                                y,
                                                X,
                                                1e-5
  );
  
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



// SPECIALIZED GRAINED INTERFACES:

// To provide specific penalties, you can also directly interface to those instead
// of using the simplified interface above. This can help you set up a more tailored
// penalization. We will use the elastic net with glmnet optimizer as an example.

// [[Rcpp::export]]
Rcpp::List elasticNet(
    const arma::colvec y, 
    arma::mat X,
    Rcpp::NumericVector startingValues,
    const arma::rowvec alpha,
    const arma::rowvec lambda
)
{
  
  // We also create a matrix which saves the parameter estimates
  // for all values of alpha and lambda. 
  Rcpp::NumericMatrix B(alpha.n_elem*lambda.n_elem, startingValues.length());
  B.fill(NA_REAL);
  
  // we also create a matrix to save the corresponding tuning parameter values
  Rcpp::NumericMatrix tpValues(alpha.n_elem*lambda.n_elem, 2);
  tpValues.fill(NA_REAL);
  Rcpp::colnames(tpValues) = Rcpp::StringVector{"alpha", "lambda"};
  // finally, let's also return the fitting function value
  Rcpp::NumericVector loss(alpha.n_elem*lambda.n_elem);
  loss.fill(NA_REAL);
  
  // now, it is time to set up the model we defined above
  
  linearRegressionModel linReg(y,X);
  
  // next, we have to define the penalties we want to use.
  // The elastic net is a combination of a ridge penalty and 
  // a lasso penalty:
  lessSEM::penaltyLASSOGlmnet lasso;
  lessSEM::penaltyRidgeGlmnet ridge;
  // Note that we used the glmnet variants of lasso and ridge. The reason
  // for this is that the glmnet implementation allows for parameter-specific
  // lambda and alpha values while the current ista implementation does not.
  
  // these penalties take tuning parameters of class tuningParametersEnetGlmnet
  lessSEM::tuningParametersEnetGlmnet tp;
  
  // finally, there is also the weights. The weights vector indicates, which
  // of the parameters is regularized (weight = 1) and which is unregularized 
  // (weight =0). It also allows for adaptive lasso weights (e.g., weight =.0123).
  // weights must be an arma::rowvec of the same length as our parameter vector.
  arma::rowvec weights(startingValues.length());
  weights.fill(1.0); // we want to regularize all parameters
  weights.at(0) = 0.0; // except for the first one, which is our intercept.
  tp.weights = weights;
  
  // if we want to fine tune the optimizer, we can use the control
  // arguments. We will start with the default control elements and 
  // tweak some arguments to our liking:
  lessSEM::controlGLMNET control = lessSEM::controlGlmnetDefault();
  
  control.breakOuter = 1e-10;
  control.breakInner = 1e-10;
  control.initialHessian = approximateHessian(startingValues,y,X,1e-10);
  
  // now it is time to iterate over all lambda and alpha values:
  int it = 0;
  for(unsigned int a = 0; a < alpha.n_elem; a++){
    for(unsigned int l = 0; l < lambda.n_elem; l++){
      
      // set the tuning parameters
      // Note: glmnet expects a vector of alphas and lambdas. However,
      // we just want to use the same value for all parameters, so we
      // will replicate the single alpha and lambda as many times as there
      // are parameters:
      tp.alpha = arma::rowvec(startingValues.length());
      tp.alpha.fill(alpha.at(a));
      tp.lambda = arma::rowvec(startingValues.length());
      tp.lambda.fill(lambda.at(l));
      
      tpValues(it,0) = alpha.at(a);
      tpValues(it,1) = lambda.at(a);
      
      // to optimize this model, we have to pass it to
      // one of the optimizers in lessSEM. These are 
      // glmnet and ista. We'll use glmnet in the following. The optimizer will
      // return an object of class fitResults which will have the following fields:
      // - convergence: boolean indicating if the convergence criterion was met (true) or not (false)
      // - fit: double with fit values
      // - fits: a vector with the fits of all iterations
      // - parameterValues the final parameter values as an arma::rowvec
      // - Hessian: the BFGS Hessian approximation
      
      lessSEM::fitResults lmFit = lessSEM::glmnet(
        linReg, // the first argument is our model
        startingValues, // the second are the parameters
        lasso, // the third is our lasso penalty
        ridge, // the fourth our ridge penalty
        tp, // the fifth is our tuning parameter 
        control // finally, let's fine tune with the control
      );
      
      loss.at(it) = lmFit.fit;
      
      for(unsigned int i = 0; i < startingValues.length(); i++){
        // let's save the parameters
        B(it,i) = lmFit.parameterValues.at(i);
        // and also carry over the current estimates for the next iteration
        startingValues.at(i) = lmFit.parameterValues.at(i);
      }
      
      // carry over Hessian for next iteration
      control.initialHessian = lmFit.Hessian;
      
      it++;
    }
  }
  
  Rcpp::List retList = Rcpp::List::create(
    Rcpp::Named("B") = B,
    Rcpp::Named("tuningParameters") = tpValues,
    Rcpp::Named("loss") = loss);
  return(
    retList
  );
}


// [[Rcpp::export]]
Rcpp::List elasticNetIsta(
    const arma::colvec y, 
    arma::mat X,
    Rcpp::NumericVector startingValues,
    const arma::rowvec alpha,
    const arma::rowvec lambda
)
{
  // IN THE FOLLOWING, WE WILL IMPLEMENT THE ELASTIC NET OPTIMIZATION WITH THE
  // ISTA OPTIMIZER. NOTE THAT THE GENERAL PROCEDURE WILL BE 
  // FAIRLY SIMILAR TO THE GLMNET OPTIMIZER DESCRIBED ABOVE.
  // THE MAIN DIFFERENCES WILL BE COMMENTED IN ALL CAPTIONS.
  
  // We have to create a matrix which saves the parameter estimates
  // for all values of alpha and lambda. 
  Rcpp::NumericMatrix B(alpha.n_elem*lambda.n_elem, startingValues.length());
  B.fill(NA_REAL);
  // we also create a matrix to save the corresponding tuning parameter values
  Rcpp::NumericMatrix tpValues(alpha.n_elem*lambda.n_elem, 2);
  tpValues.fill(NA_REAL);
  Rcpp::colnames(tpValues) = Rcpp::StringVector{"alpha", "lambda"};
  // finally, let's also return the fitting function value
  Rcpp::NumericVector loss(alpha.n_elem*lambda.n_elem);
  loss.fill(NA_REAL);
  
  // now, it is time to set up the model we defined above
  
  linearRegressionModel linReg(y,X);
  
  // next, we have to define the penalties we want to use.
  // The elastic net is a combination of a ridge penalty and 
  // a lasso penalty. 
  // NOTE: HERE COMES THE BIGGEST DIFFERENCE BETWEEN GLMNET AND ISTA:
  // 1) ISTA ALSO REQUIRES THE DEFINITION OF A PROXIMAL OPERATOR. THESE
  //    ARE CALLED proximalOperatorZZZ IN lessSEM (e.g., proximalOperatorLasso 
  //    for lasso).
  // 2) THE SMOOTH PENALTY (RIDGE) AND THE LASSO PENALTY MUST HAVE SEPARATE 
  //    TUNING PARMAMETERS.
  lessSEM::proximalOperatorLasso proxOp; // HERE, WE DEFINE THE PROXIMAL OPERATOR
  lessSEM::penaltyLASSO lasso; 
  lessSEM::penaltyRidge ridge;
  // BOTH, LASSO AND RIDGE take tuning parameters of class tuningParametersEnet
  lessSEM::tuningParametersEnet tpLasso;
  lessSEM::tuningParametersEnet tpRidge;
  
  // finally, there is also the weights. The weights vector indicates, which
  // of the parameters is regularized (weight = 1) and which is unregularized 
  // (weight =0). It also allows for adaptive lasso weights (e.g., weight =.0123).
  // weights must be an arma::rowvec of the same length as our parameter vector.
  arma::rowvec weights(startingValues.length());
  weights.fill(1.0); // we want to regularize all parameters
  weights.at(0) = 0.0; // except for the first one, which is our intercept.
  tpLasso.weights = weights;
  tpRidge.weights = weights;
  
  // if we want to fine tune the optimizer, we can use the control
  // arguments. We will start with the default control elements and 
  // tweak some arguments to our liking.
  // THE CONTROL ELEMENT IS CALLED DIFFERENTLY FROM GLMNET:
  lessSEM::control control = lessSEM::controlDefault();
  control.breakOuter = 1e-10;
  
  // now it is time to iterate over all lambda and alpha values:
  int it = 0;
  for(unsigned int a = 0; a < alpha.n_elem; a++){
    for(unsigned int l = 0; l < lambda.n_elem; l++){
      
      // DON'T FORGET TO UPDATE BOTH TUNING PARAMETERS
      // set the tuning parameters
      tpLasso.alpha = alpha.at(a);
      tpLasso.lambda = lambda.at(l);
      tpRidge.alpha = alpha.at(a);
      tpRidge.lambda = lambda.at(l);
      
      tpValues(it,0) = alpha.at(a);
      tpValues(it,1) = lambda.at(a);
      
      // to optimize this model, we have to pass it to
      // one of the optimizers in lessSEM. These are 
      // glmnet and ista. We'll use glmnet in the following. The optimizer will
      // return an object of class fitResults which will have the following fields:
      // - convergence: boolean indicating if the convergence criterion was met (true) or not (false)
      // - fit: double with fit values
      // - fits: a vector with the fits of all iterations
      // - parameterValues the final parameter values as an arma::rowvec
      
      lessSEM::fitResults lmFit = lessSEM::ista(
        linReg, // the first argument is our model
        startingValues, // the second are the parameters
        proxOp,
        lasso, // the third is our lasso penalty
        ridge, // the fourth our ridge penalty
        tpLasso, // the fifth is our tuning parameter FOR THE LASSO 
        tpRidge, // THE SIXTH IS OUR TUNING PARAMETER FOR THE RIDGE PENALTY
        control // finally, let's fine tune with the control
      );
      
      loss.at(it) = lmFit.fit;
      
      for(unsigned int i = 0; i < startingValues.length(); i++){
        // let's save the parameters
        B(it,i) = lmFit.parameterValues.at(i);
        // and also carry over the current estimates for the next iteration
        startingValues.at(i) = lmFit.parameterValues.at(i);
      }
      
      it++;
    }
  }
  
  Rcpp::List retList = Rcpp::List::create(
    Rcpp::Named("B") = B,
    Rcpp::Named("tuningParameters") = tpValues,
    Rcpp::Named("loss") = loss);
  return(
    retList
  );
}


// [[Rcpp::export]]
Rcpp::List scadIsta(
    const arma::colvec y, 
    arma::mat X,
    Rcpp::NumericVector startingValues,
    const arma::rowvec theta,
    const arma::rowvec lambda
)
{
  // IN THE FOLLOWING, WE WILL IMPLEMENT THE SCAD OPTIMIZATION WITH THE
  // ISTA OPTIMIZER. NOTE THAT THE GENERAL PROCEDURE WILL BE 
  // FAIRLY SIMILAR TO THE ELASTIC NET OPTIMIZATION WITH ISTA DESCRIBED ABOVE.
  // THE MAIN DIFFERENCES WILL BE COMMENTED IN ALL CAPTIONS.
  
  // We have to create a matrix which saves the parameter estimates
  // for all values of alpha and lambda. 
  Rcpp::NumericMatrix B(theta.n_elem*lambda.n_elem, startingValues.length());
  B.fill(NA_REAL);
  // we also create a matrix to save the corresponding tuning parameter values
  Rcpp::NumericMatrix tpValues(theta.n_elem*lambda.n_elem, 2);
  tpValues.fill(NA_REAL);
  Rcpp::colnames(tpValues) = Rcpp::StringVector{"theta", "lambda"};
  // finally, let's also return the fitting function value
  Rcpp::NumericVector loss(theta.n_elem*lambda.n_elem);
  loss.fill(NA_REAL);
  
  // now, it is time to set up the model we defined above
  
  linearRegressionModel linReg(y,X);
  
  // next, we have to define the penalties we want to use.
  // The elastic net is a combination of a ridge penalty and 
  // a lasso penalty. 
  // NOTE: HERE COMES THE BIGGEST DIFFERENCE BETWEEN GLMNET AND ISTA:
  // 1) ISTA ALSO REQUIRES THE DEFINITION OF A PROXIMAL OPERATOR. THESE
  //    ARE CALLED proximalOperatorZZZ IN lessSEM (e.g., proximalOperatorLasso 
  //    for lasso).
  // 2) THE SMOOTH PENALTY (RIDGE) AND THE LASSO PENALTY MUST HAVE SEPARATE 
  //    TUNING PARMAMETERS.
  // FOR THE SCAD PENALTY, WE MUST REPLACE THE PROXIMAL OPERATOR AND 
  // THE PENALTY FUNCTION WITH THOSE OF THE SCAD
  lessSEM::proximalOperatorScad proxOpScad; // HERE, WE DEFINE THE PROXIMAL OPERATOR
  lessSEM::penaltyScad scad;  // HERE, WE DEFINE THE SCAD PENALTY FUNCTION
  
  // LET'S ALSO DEFINE THE TUNING PARAMETERS. THESE ARE NOW GIVEN BY
  lessSEM::tuningParametersScad tpScad;
  
  // THE SCAD PENALTY DOES NOT HAVE A DIFFERENTIABLE PENALTY PART. THEREFORE,
  // WE DO NOT USE RIDGE HERE. HOWEVER, ISTA STILL NEEDS A DIFFERENTIABLE
  // PENALTY OBJECT. TO ACCOUNT FOR THIS, DEFINE A DIFFERENTIABLE PENALTY OF
  // TYPE noSmoothPenalty. WE MUST ALSO DEFINE TUNING PARAMETERS FOR THIS
  // PENALTY; JUST USE THE SAME ONES AS FOR THE SCAD: lessSEM::tuningParametersScad
  lessSEM::noSmoothPenalty<lessSEM::tuningParametersScad> noSmoothP;
  // BOTH, LASSO AND RIDGE take tuning parameters of class tuningParametersEnet
  lessSEM::tuningParametersScad tpNoSmoothP;
  tpNoSmoothP.lambda = 0.0; // JUST TO BE SURE, WE CAN ALSO DEACTIVATE ANY SMOOTH
  // PENALTY BY SETTING THE TUNING PARAMETER LAMBDA TO 0
  
  // finally, there is also the weights. The weights vector indicates, which
  // of the parameters is regularized (weight = 1) and which is unregularized 
  // (weight =0). It also allows for adaptive lasso weights (e.g., weight =.0123).
  // weights must be an arma::rowvec of the same length as our parameter vector.
  arma::rowvec weights(startingValues.length());
  weights.fill(1.0); // we want to regularize all parameters
  weights.at(0) = 0.0; // except for the first one, which is our intercept.
  tpScad.weights = weights;
  tpNoSmoothP.weights = weights;
  
  // if we want to fine tune the optimizer, we can use the control
  // arguments. We will start with the default control elements and 
  // tweak some arguments to our liking.
  // THE CONTROL ELEMENT IS CALLED DIFFERENTLY FROM GLMNET:
  lessSEM::control control = lessSEM::controlDefault();
  control.breakOuter = 1e-10;
  
  // now it is time to iterate over all lambda and theta values:
  int it = 0;
  for(unsigned int a = 0; a < theta.n_elem; a++){
    for(unsigned int l = 0; l < lambda.n_elem; l++){
      
      // DON'T FORGET TO UPDATE BOTH TUNING PARAMETERS
      // set the tuning parameters
      tpScad.theta = theta.at(a);
      tpScad.lambda = lambda.at(l);
      
      tpValues(it,0) = theta.at(a);
      tpValues(it,1) = lambda.at(a);
      
      // to optimize this model, we have to pass it to
      // one of the optimizers in lessSEM. These are 
      // glmnet and ista. We'll use glmnet in the following. The optimizer will
      // return an object of class fitResults which will have the following fields:
      // - convergence: boolean indicating if the convergence criterion was met (true) or not (false)
      // - fit: double with fit values
      // - fits: a vector with the fits of all iterations
      // - parameterValues the final parameter values as an arma::rowvec
      
      lessSEM::fitResults lmFit = lessSEM::ista(
        linReg, // the first argument is our model
        startingValues, // the second are the parameters
        proxOpScad,
        scad, // the third is our scad penalty
        noSmoothP, // the fourth our smooth penalty
        tpScad, // the fifth is our tuning parameter FOR THE LASSO 
        tpNoSmoothP, // THE SIXTH IS OUR TUNING PARAMETER FOR THE RIDGE PENALTY
        control // finally, let's fine tune with the control
      );
      
      loss.at(it) = lmFit.fit;
      
      for(unsigned int i = 0; i < startingValues.length(); i++){
        // let's save the parameters
        B(it,i) = lmFit.parameterValues.at(i);
        // and also carry over the current estimates for the next iteration
        startingValues.at(i) = lmFit.parameterValues.at(i);
      }
      
      it++;
    }
  }
  
  Rcpp::List retList = Rcpp::List::create(
    Rcpp::Named("B") = B,
    Rcpp::Named("tuningParameters") = tpValues,
    Rcpp::Named("loss") = loss);
  return(
    retList
  );
}

