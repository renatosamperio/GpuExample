
#include "g_simulate.h"

__global__ void vectorAdd(const double *dCounterparty, 
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

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dCounterparty, 
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
    return 1;
}