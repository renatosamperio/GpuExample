#include "xorshift_generator.h"

// CUDA kernel to generate random numbers using Xorshift
__global__ void xorshiftRandomKernel(unsigned int* random_numbers, int num_elements, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = XORSHIFT_SEED + tid; // Seed initialization, unique for each thread
    
    for (int i = 0; i < num_elements; i++) {
        int idx = tid * num_elements + i;
        if (idx < N) {
            x ^= (x << XORSHIFT_A);
            x ^= (x >> XORSHIFT_B);
            x ^= (x << XORSHIFT_C);
            random_numbers[tid * num_elements + i] = static_cast<int>(x);
        }
        else break;
    }
}

void get_random_numbers(unsigned int* host_random_numbers, int N) {
    // Error code to check return values for CUDA calls
    // cudaError_t err = cudaSuccess;

    // int num_elements = 1000; // Number of random numbers to generate per thread

    // Maximum number of threads per block: 1024
    // Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    // This two numbers should fit in 1024
    int num_threads = 256;  // Number of threads in each block
    int num_blocks = 4;     // Number of blocks
    int num_elements = ceil( N / (num_threads * num_blocks));
    Rcpp::Rcout << "Processing max elements per thread" << num_elements <<"\n";

    size_t random_size = N * sizeof(unsigned int);
    unsigned int* device_random_numbers;
    err = cudaMalloc((void**)&device_random_numbers, random_size);

    // Launch the kernel
    xorshiftRandomKernel<<<num_blocks, num_threads>>>(device_random_numbers, num_elements, N);
    
    // Copy the results back to the host
    err = cudaMemcpy(host_random_numbers, device_random_numbers, random_size, cudaMemcpyDeviceToHost);

    // Cleanup
    err = cudaFree(device_random_numbers);

    // Print a few random numbers from the first thread
    for (int i = 0; i < 10; i++) {
      Rcpp::Rcout << "Array ["<< i<< "]: " << host_random_numbers[i] << "\n";
    }

}


Rcpp::IntegerVector xorshift_generator(Rcpp::IntegerVector numbers, int N) {
   
    Rcpp::Rcout << "Calculating "<< N << " random numbers" <<"\n";
    // Create array for random numbers
    unsigned int* host_random_numbers = new unsigned int[N];

    // Generate random numbers
    get_random_numbers(host_random_numbers, N);

    // Passing data from array into an vector
    std::vector<unsigned int> numbers_as_std_vector(N);
    memcpy(&numbers_as_std_vector[0], 
            &host_random_numbers[0], N*sizeof(unsigned int));

    // Print a few random numbers from the first thread
    for (int i = 0; i < 10; i++) {
      Rcpp::Rcout << "Vector ["<< i<< "]: " << numbers_as_std_vector[i] << "\n";
    }

    // Clean up
    delete[] host_random_numbers;

    return Rcpp::IntegerVector(numbers_as_std_vector.begin(), 
                                numbers_as_std_vector.end());
}
