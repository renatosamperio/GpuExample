#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <curand_kernel.h>

//' Runs GPU-based simulator
//' 
//' @param portfolio `matrix` porfolio
//' @param n_factor `integer` number of factors
//' @param n_sim `integer` number of simulations
//' @export
// [[Rcpp::export]]
int xorshift_generator(Rcpp::NumericVector numbers, int N );
