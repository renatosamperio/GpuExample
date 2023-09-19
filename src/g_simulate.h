
#include <Rcpp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <cuda_runtime.h>

//' Runs GPU-based simulator
//' 
//' @param portfolio `matrix` porfolio
//' @param n_factor `integer` number of factors
//' @param n_sim `integer` number of simulations
//' @export
// [[Rcpp::export]]
int g_simulate(Rcpp::NumericMatrix portfolio,  int n_factor, int n_sim );
