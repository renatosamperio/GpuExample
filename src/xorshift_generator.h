#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <curand_kernel.h>

//' Runs GPU-based simulator
//' 
//' @param numbers `vector` input vector
//' @param N `integer` number of samples
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector xorshift_generator(Rcpp::NumericVector numbers, int N );
