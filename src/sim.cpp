#include "sim.h"
#include "sim_utils.h"

NumericVector sim(NumericMatrix portfolio, int n_factor, int n_sim) {
   
   // hard coded factor loadings
   const float w_global = sqrt(.15);
   const float w_counterparty = sqrt(.65); // counterparty loading
   const float w_country = sqrt(.1);  // country loading
   const float w_industry = sqrt(.1); // industry loading
   
   const int n_row = portfolio.nrow();
   NumericVector losses(n_sim);
   std::vector<float> factors(n_factor);
   unsigned int seed = 13452452;  // hard coded seed
   
   for (int i_sim=0; i_sim<n_sim; i_sim++) {
      get_normal_variates(seed, factors, n_factor);
      float loss=0;
      for (int i_row=0; i_row<n_row; i_row++) {
         float r_tot = w_global * factors[0] + w_counterparty * factors[portfolio(i_row, 0)] +
            w_country * factors[portfolio(i_row, 1)] + w_industry * factors[portfolio(i_row, 2)];
         if (r_tot < portfolio(i_row, 4)) {
            loss += portfolio(i_row, 3);
         }
      }
      losses(i_sim) = loss;
   }
   return(losses);
}

