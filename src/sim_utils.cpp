#include "sim_utils.h"
#include <math.h>

unsigned int rand_xorshift(unsigned int rng_state) {
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

void get_normal_variates(unsigned int& x, float* variates, int n) {
   int m = n / 2;
   for (int i=0; i<m; i++) {
      x = rand_xorshift(x);
      float u1 = (float) x / (float) 0xFFFFFFFF;
      x = rand_xorshift(x);
      float u2 = (float) x / (float) 0xFFFFFFFF;
      float phi = 2*M_PI*u1;
      float r = sqrt(-2*log(u2));
      variates[2*i] = r*cos(phi);
      variates[2*i+1] = r*sin(phi);
   }
   if (2*m<n) {
      x = rand_xorshift(x);
      float u1 = (float) x / (float) 0xFFFFFFFF;
      x = rand_xorshift(x);
      float u2 = (float) x / (float) 0xFFFFFFFF;
      float phi = 2*M_PI*u1;
      float r = sqrt(-2*log(u2));
      variates[n-1] = r*cos(phi);
   }
}

void get_normal_variates(unsigned int& x, std::vector<float>& variates, int n) {
   int m = n/2;
   for (int i=0; i<m; i++) {
      x = rand_xorshift(x);
      float u1 = (float) x / (float) 0xFFFFFFFF;
      x = rand_xorshift(x);
      float u2 = (float) x / (float) 0xFFFFFFFF;
      float phi = 2*M_PI*u1;
      float r = sqrt(-2*log(u2));
      variates[2*i] = r*cos(phi);
      variates[2*i+1] = r*sin(phi);
   }
   if (2*m<n) {
      x = rand_xorshift(x);
      float u1 = (float) x / (float) 0xFFFFFFFF;
      x = rand_xorshift(x);
      float u2 = (float) x / (float) 0xFFFFFFFF;
      float phi = 2*M_PI*u1;
      float r = sqrt(-2*log(u2));
      variates[n-1] = r*cos(phi);
   }
}