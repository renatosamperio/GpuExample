#include "simulate.h"

#define THREAD_NUM 1024

__device__ unsigned int rand_xorshift_gpu(unsigned int rng_state){
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

// get_normal_variates(seed_thread, factors, n_factor);
__device__ void get_normal_variates(unsigned int x, 
                         float* variates, 
                         int n) {
   int m = n/2;
   for (int i=0; i<m; i++) {
      x = rand_xorshift_gpu(x);
      float u1 = (float) x / (float) 0xFFFFFFFF;
      x = rand_xorshift_gpu(x);
      float u2 = (float) x / (float) 0xFFFFFFFF;
      float phi = 2*M_PI*u1;
      float r = sqrt(-2*log(u2));
      variates[2*i] = r*cos(phi);
      variates[2*i+1] = r*sin(phi);
   }
   if (2*m<n) {
      x = rand_xorshift_gpu(x);
      float u1 = (float) x / (float) 0xFFFFFFFF;
      x = rand_xorshift_gpu(x);
      float u2 = (float) x / (float) 0xFFFFFFFF;
      float phi = 2*M_PI*u1;
      float r = sqrt(-2*log(u2));
      variates[n-1] = r*cos(phi);
   }
}

__global__ void simulate(
    unsigned int* seed,             // 0
    float* result,                  // 1
    unsigned int* counterparty,     // 2
    unsigned int* country,          // 3
    unsigned int* industry,         // 4
    float* exposure,                // 5
    float* pd_th,                   // 6
    int n_row,                      // 7
    float* factors_buffer,          // 8
    int n_factor,                   // 9
    int n_sim_per_thread            // 10
) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;

   float w_global = sqrt(.15);
   float w_counterparty = sqrt(.65); // counterparty loading
   float w_country = sqrt(.1);  // country loading
   float w_industry = sqrt(.1); // industry loading

   float* factors = &factors_buffer[index * n_factor];
   unsigned int seed_thread = seed[index];

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

//' Runs GPU-based simulator
//' 
//' @param portfolio `matrix` porfolio
//' @param n_factor `integer` number of factors
//' @param n_sim `integer` number of simulations
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector simulator(
   Rcpp::NumericMatrix portfolio,  int n_factor, int n_sim ) {

   // Error code to check return values for CUDA calls
   cudaError_t err = cudaSuccess;

   // Initialising device memory pointers
   unsigned int *dCounterparty = NULL;
   unsigned int *dCountry      = NULL;
   unsigned int *dIndustry     = NULL;
   float        *dExposure     = NULL;
   float        *dPdth         = NULL;

   int n_row = portfolio.nrow();
   int n_cols = portfolio.ncol();
   // size_t hSize = n_row * sizeof(unsigned int);
   Rcpp::Rcout << "Matrix size ("<< n_row<<", "<<n_cols<<"): " 
               << n_factor << " : " << n_sim
               << std::endl;

   // Expand matrix into Rcpp vectors per column
   Rcpp::NumericVector counterparty = portfolio( Rcpp::_ , 0 );
   Rcpp::NumericVector country      = portfolio( Rcpp::_ , 1 );
   Rcpp::NumericVector industry     = portfolio( Rcpp::_ , 2 );
   Rcpp::NumericVector exposure     = portfolio( Rcpp::_ , 3 );
   Rcpp::NumericVector pdTh         = portfolio( Rcpp::_ , 4 );

   // Converting Rcpp to std vector
   std::vector<unsigned int> vCounterparty(counterparty.begin(), counterparty.end());
   std::vector<unsigned int> vCountry(country.begin(), country.end());
   std::vector<unsigned int> vIndustry(industry.begin(), industry.end());
   std::vector<unsigned int> vExposure(exposure.begin(), exposure.end());
   std::vector<unsigned int> vPdth(pdTh.begin(), pdTh.end());

   // Casting std vector into a pointer
   unsigned int* pCounterparty = vCounterparty.data();
   unsigned int* pCountry = vCountry.data();
   unsigned int* pIndustry = vIndustry.data();
   unsigned int* pExposure = vExposure.data();
   unsigned int* pPdth = vPdth.data();

   // Copy input data from host to device
   err = cudaMalloc((void **)&dCounterparty, n_row);
   err = cudaMalloc((void **)&dCountry, n_row);
   err = cudaMalloc((void **)&dIndustry, n_row);
   err = cudaMalloc((void **)&dExposure, n_row);
   err = cudaMalloc((void **)&dPdth, n_row);

   err = cudaMemcpy(dCounterparty, pCounterparty, n_row, cudaMemcpyHostToDevice);
   err = cudaMemcpy(dCountry, pCountry, n_row, cudaMemcpyHostToDevice);
   err = cudaMemcpy(dIndustry, pIndustry, n_row, cudaMemcpyHostToDevice);
   err = cudaMemcpy(dExposure, pExposure, n_row, cudaMemcpyHostToDevice);
   err = cudaMemcpy(dPdth, pPdth, n_row, cudaMemcpyHostToDevice);

   // Determine the optimal number of threads per block and blocks per grid
   int threadsPerBlock = THREAD_NUM;
   int blocksPerGrid = (n_row + threadsPerBlock - 1) / threadsPerBlock;

   // Define simulator variables
   unsigned int *dseeds         = NULL;
   float       *dresult         = NULL;
   float       *dfactors_buffer = NULL;

   // Allocating host memory
   float       * hresults       = (float *)malloc(THREAD_NUM);
  
   // Prepare simulator variables
   err = cudaMalloc((void **)&dseeds,  THREAD_NUM);
   err = cudaMalloc((void **)&dresult, THREAD_NUM);
   err = cudaMalloc((void **)&dfactors_buffer, THREAD_NUM * n_factor);

   // Launch the Vector Add CUDA Kernel
   simulate<<<blocksPerGrid, threadsPerBlock>>>(
      dseeds,
      dresult,
      dCounterparty,
      dCountry,
      dIndustry,
      dExposure,
      dPdth,
      n_row,
      dfactors_buffer,
      n_factor,
      THREAD_NUM
   );

   // confirm kernel returned results
   err = cudaGetLastError();
   if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
               cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }

   // pass data from device into host memory space
   err = cudaMemcpy(hresults, dresult, THREAD_NUM, cudaMemcpyDeviceToHost);

   // pass data into a vector and then into R
   std::vector<int> std_result(hresults, hresults + sizeof hresults / sizeof hresults[0]);
   Rcpp::NumericVector r_result(std_result.begin(), std_result.end());
   
   // release device and host memory 
   cudaFree(dCounterparty);
   cudaFree(dCountry);
   cudaFree(dIndustry);
   cudaFree(dExposure);
   cudaFree(dPdth);
   free(hresults);
   
   return r_result;
}
