
#include "g_simulate.h"

int g_simulate(Rcpp::NumericMatrix portfolio,  int n_factor, int n_sim ) {
    
    int numRows = portfolio.nrow();
    int numCols = portfolio.ncol();
    Rcpp::Rcout << "Matrix size ("<< numRows<<", "<<numCols<<"): " 
                << n_factor << " : " << n_sim
                << std::endl;

    Rcpp::NumericVector counterparty = portfolio( Rcpp::_ , 0 );
    Rcpp::NumericVector country      = portfolio( Rcpp::_ , 1 );
    Rcpp::NumericVector industry     = portfolio( Rcpp::_ , 2 );
    Rcpp::NumericVector exposure     = portfolio( Rcpp::_ , 3 );
    Rcpp::NumericVector pd_th        = portfolio( Rcpp::_ , 4 );
    return 1;
}