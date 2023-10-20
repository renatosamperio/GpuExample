#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <curand_kernel.h>

// Xorshift parameters (you may customize these)
#define XORSHIFT_SEED 123456789
#define XORSHIFT_A 13
#define XORSHIFT_B 17
#define XORSHIFT_C 5

// Wrapper to calculate random numbers
void get_random_numbers(unsigned int* host_random_numbers);

//' Runs GPU-based simulator
//' 
//' @param numbers `vector` input vector
//' @param N `integer` number of samples
//' @export
// [[Rcpp::export]]
Rcpp::IntegerVector xorshift_generator(Rcpp::IntegerVector numbers, int N );
