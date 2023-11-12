#include <cmath>

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
   }
}
