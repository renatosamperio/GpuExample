#include <math.h>
#include <Rcpp.h>

//' Runs GPU-based simulator
//' 
//' @param portfolio `matrix` porfolio
//' @param n_factor `integer` number of factors
//' @param n_sim `integer` number of simulations
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector simulator(Rcpp::NumericMatrix portfolio,  int n_factor, int n_sim );
