#include <math.h>

unsigned int rand_xorshift_gpu(unsigned int rng_state);
void get_normal_variates(unsigned int x, 
                         float* variates, 
                         int n);

void simulate(
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
    int n_sim_per_thread           // 10
) ;