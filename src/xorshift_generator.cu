#include "xorshift_generator.h"

// Xorshift random number generator kernel
__global__ void xorshift_kernel(unsigned int seed, unsigned int* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = seed + idx; // Use unique seed for each thread

    for (int i = 0; i < N; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        output[idx * N + i] = x;
    }
}

int xorshift_generator(Rcpp::NumericVector numbers, int N) {
   
    std::vector<double> numbers_as_std_vector = Rcpp::as<std::vector<double> >(numbers); 

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // const int N = 100; // Number of random numbers to generate per thread
    const int numThreads = 256; // Number of CUDA threads per block
    const int numBlocks = 4;   // Number of CUDA blocks
    size_t hSize = N * sizeof(double); // size of output vector

    // Allocating host memory
    double* h_numbers = (double *)malloc(hSize);
    unsigned int* d_output; // Device buffer for random numbers
    unsigned int* h_output = new unsigned int[numThreads * numBlocks * N]; // Host buffer for random numbers

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_output, sizeof(unsigned int) * numThreads * numBlocks * N);

    // Launch the Xorshift kernel
    xorshift_kernel<<<numBlocks, numThreads>>>(12345, d_output, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the random numbers from device to host
    err = cudaMemcpy(h_output, d_output, sizeof(unsigned int) * numThreads * numBlocks * N, cudaMemcpyDeviceToHost);

    // Print the generated random numbers (for demonstration purposes)
    for (int i = 0; i < numThreads * numBlocks * N; i++) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % 10 == 0) {
            std::cout << std::endl;
        }
    }

    // Clean up
    delete[] h_output;
    cudaFree(d_output);

    return 0;
}
