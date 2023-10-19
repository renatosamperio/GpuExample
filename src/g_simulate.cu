
#include "g_simulate.h"

// Documented value on page 22. 
// https://arxiv.org/pdf/1903.07486.pdf#:~:text=32%2C%20as%20per%20public%20NVidia,by%20just%20increasing%20block%20count.
#define THREAD_NUM 1024
// hard coded factor loadings
const float w_global = sqrt(.15);
const float w_counterparty = sqrt(.65); // counterparty loading
const float w_country = sqrt(.1);  // country loading
const float w_industry = sqrt(.1); // industry loading

unsigned int rand_xorshift(unsigned int rng_state) {
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

__global__ void g_simulate(const double *dCounterparty, 
                          const double *dCountry, 
                          const double *dIndustry, 
                          const double *dExposure, 
                          const double *dPdth, 
                          double *dLosses,
                          int numRows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numRows) {
        dLosses[i] = dCounterparty[i] + dCountry[i] + dIndustry[i] + dExposure[i] + dPdth[i] ;
    }

   device float* factors = &factors_buffer[index * n_factor];
   
   uint seed_thread = seed[index];
   
   for (int i_sim=0; i_sim<n_sim_per_thread; i_sim++) {
      get_normal_variates(seed_thread, factors, n_factor);
      float loss=0;
      for (int i=0; i<n_row; i++) {
         float r_tot = w_global * factors[0] + w_counterparty * factors[counterparty[i]] +
                       w_country * factors[country[i]] + w_industry * factors[industry[i]];
         if (r_tot < pd_th[i]) {
            loss += exposure[i];
         }
      }
      result[index*n_sim_per_thread+i_sim] = loss;
   }
   
   
}


int g_simulate(Rcpp::NumericMatrix portfolio,  int n_factor, int n_sim ) {
    
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Initialising device memory pointers
    double *dCounterparty = NULL;
    double *dCountry      = NULL;
    double *dIndustry     = NULL;
    double *dExposure     = NULL;
    double *dPdth         = NULL;
    double *dLosses       = NULL;
  
    int numRows = portfolio.nrow();
    int numCols = portfolio.ncol();
    size_t hSize = numRows * sizeof(double);
    Rcpp::Rcout << "Matrix size ("<< numRows<<", "<<numCols<<"): " 
                << n_factor << " : " << n_sim
                << std::endl;

    // Expand matrix into Rcpp vectors per column
    Rcpp::NumericVector counterparty = portfolio( Rcpp::_ , 0 );
    Rcpp::NumericVector country      = portfolio( Rcpp::_ , 1 );
    Rcpp::NumericVector industry     = portfolio( Rcpp::_ , 2 );
    Rcpp::NumericVector exposure     = portfolio( Rcpp::_ , 3 );
    Rcpp::NumericVector pdTh        = portfolio( Rcpp::_ , 4 );

    // Converting Rcpp to std vector
    std::vector<double> vCounterparty(counterparty.begin(), counterparty.end());
    std::vector<double> vCountry(country.begin(), country.end());
    std::vector<double> vIndustry(industry.begin(), industry.end());
    std::vector<double> vExposure(exposure.begin(), exposure.end());
    std::vector<double> vPdth(pdTh.begin(), pdTh.end());

    // Casting std vector into a pointer
    double* pCounterparty = vCounterparty.data();
    double* pCountry = vCountry.data();
    double* pIndustry = vIndustry.data();
    double* pExposure = vExposure.data();
    double* pPdth = vPdth.data();

    // Allocating host memory
    double* hLosses = (double *)malloc(hSize);

    // Copy input data from host to device
    err = cudaMalloc((void **)&dCounterparty, numRows);
    err = cudaMalloc((void **)&dCountry, numRows);
    err = cudaMalloc((void **)&dIndustry, numRows);
    err = cudaMalloc((void **)&dExposure, numRows);
    err = cudaMalloc((void **)&dPdth, numRows);
    err = cudaMalloc((void **)&dLosses, numRows);

    err = cudaMemcpy(dCounterparty, pCounterparty, numRows, cudaMemcpyHostToDevice);
    err = cudaMemcpy(dCountry, pCountry, numRows, cudaMemcpyHostToDevice);
    err = cudaMemcpy(dIndustry, pIndustry, numRows, cudaMemcpyHostToDevice);
    err = cudaMemcpy(dExposure, pExposure, numRows, cudaMemcpyHostToDevice);
    err = cudaMemcpy(dPdth, pPdth, numRows, cudaMemcpyHostToDevice);

    // Determine the optimal number of threads per block and blocks per grid
    int threadsPerBlock = THREAD_NUM;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the Vector Add CUDA Kernel
    randomize<<<blocksPerGrid, threadsPerBlock>>>(dCounterparty, 
                                                  dCountry,
                                                  dIndustry,
                                                  dExposure,
                                                  dPdth,
                                                  dLosses,
                                                  numRows);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMemcpy(hLosses, dLosses, hSize, cudaMemcpyDeviceToHost);
    for (int i=0; i<5; i++) {
        Rcpp::Rcout << "  hLosses["<< i <<"] = "<<hLosses[i]<<"): " << std::endl;
    }
    free(hLosses);
    return 1;
}