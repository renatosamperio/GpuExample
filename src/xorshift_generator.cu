#include "xorshift_generator.h"

// CUDA kernel to generate random numbers using Xorshift
__global__ void xorshiftRandomKernel(unsigned int* random_numbers, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = XORSHIFT_SEED + tid; // Seed initialization, unique for each thread
    
    for (int i = 0; i < num_elements; i++) {
        x ^= (x << XORSHIFT_A);
        x ^= (x >> XORSHIFT_B);
        x ^= (x << XORSHIFT_C);
        random_numbers[tid * num_elements + i] = x;
    }
}

void get_random_numbers(unsigned int* host_random_numbers) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int num_elements = 1000; // Number of random numbers to generate per thread
    int num_threads = 128;  // Number of threads in each block
    int num_blocks = 4;    // Number of blocks

    int total_elements = num_threads * num_blocks * num_elements;
    size_t random_size = total_elements * sizeof(unsigned int);
    
    // unsigned int* host_random_numbers = new unsigned int[total_elements];
    unsigned int* device_random_numbers;

    err = cudaMalloc((void**)&device_random_numbers, random_size);

    // Launch the kernel
    xorshiftRandomKernel<<<num_blocks, num_threads>>>(device_random_numbers, num_elements);
    
    // Copy the results back to the host
    err = cudaMemcpy(host_random_numbers, device_random_numbers, random_size, cudaMemcpyDeviceToHost);

    // Cleanup
    err = cudaFree(device_random_numbers);

    // Print a few random numbers from the first thread
    for (int i = 0; i < 10; i++) {
        // std::cout << "Array Number " << i << ": " << host_random_numbers[i] << std::endl;
      printf("Array number[%d] = %u \n", i, host_random_numbers[i]);
    }

    // delete[] host_random_numbers;

}


Rcpp::IntegerVector xorshift_generator(Rcpp::IntegerVector numbers, int N) {
   
    // Create array for random numbers
    unsigned int* host_random_numbers = new unsigned int[N];

    // Generate random numbers
    get_random_numbers(host_random_numbers);

    // Passing data from array into an vector
    std::vector<unsigned int> numbers_as_std_vector(N);
    memcpy(&numbers_as_std_vector[0], 
            &host_random_numbers[0], N*sizeof(unsigned int));

    // Print a few random numbers from the first thread
    for (int i = 0; i < 10; i++) {
      printf("Vector number[%d] = %u \n", i, numbers_as_std_vector[i]);
        // std::cout << "Vector Number " << i << ": " << numbers_as_std_vector[i] << std::endl;
    }

    // Clean up
    delete[] host_random_numbers;

    return Rcpp::IntegerVector(numbers_as_std_vector.begin(), 
                                numbers_as_std_vector.end());
}
