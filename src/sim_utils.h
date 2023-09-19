// Utilities for simulator
#include <vector>

unsigned int rand_xorshift(unsigned int rng_state);
void get_normal_variates(unsigned int& x, float* variates, int n);
void get_normal_variates(unsigned int& x, std::vector<float>& variates, int n);
