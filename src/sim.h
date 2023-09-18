
#include <vector>
#include <Rcpp.h>
#include <random>
#include <math.h>

using namespace Rcpp;

int rand_xorshift(uint rng_state);
void get_normal_variates(uint& x, std::vector<float>& variates, const int n);

//' Runs a portfolio simulation in C++
//' 
//' @param portfolio `matrix` porfolio
//' @param n_factor `integer` number of factors
//' @param n_sim `integer` number of simulations
// [[Rcpp::export]]
NumericVector sim(NumericMatrix portfolio,  int n_factor, int n_sim);