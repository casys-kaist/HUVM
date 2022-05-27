/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <string.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "common.h"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(int argc, char* argv[]) {

  if(argc != 4) {
    printf("Usage: ./vectorAdd [memory_overcommitment (L or M or H)] [num_threads] [num_iterations]\n");
    return -1;
  }

  printf("num threads: %d\n", atoi(argv[2]));
  printf("num iterations: %d\n", atoi(argv[3]));

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  //size_t numElements = (size_t)BLOCK_SIZE * 3000;
  size_t numElements = 0;
  int num_iter = atoi(argv[3]);

  // Print the vector length to be used, and compute its size
  if (strcmp(argv[1], "L") == 0) {
    numElements = 1500000000;
  } else if (strcmp(argv[1], "M") == 0) {
    numElements = 1800000000;
  } else if (strcmp(argv[1], "H") == 0) {
    numElements = 2000000000;
  } else {
    fprintf(stderr, "memory overcommitment L or M or H\n");
    return -1;
  }

  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %lu elements]\n", numElements);

  // Allocate the device input vector A
  float *d_A, *d_B, *d_C;

  err = cudaMallocManaged((void**)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  //float *d_B;
  err = cudaMallocManaged((void**)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  //float *d_C;
  err = cudaMallocManaged((void**)&d_C, size);


  if (num_iter == 1) { // Test only when 1 iteration
      // Initialize the host input vectors
      for (int i = 0; i < numElements; ++i) {
        d_A[i] = rand() / (float)RAND_MAX;
        d_B[i] = rand() / (float)RAND_MAX;
        d_C[i] = 0;
      }
  }

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = atoi(argv[2]);
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);


  for (int i = 0; i < num_iter; i++) {
      vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
      cudaDeviceSynchronize();
  }

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  if (num_iter == 1) {
      // Verify that the result vector is correct (Only test when 1 iteration)
      for (int i = 0; i < numElements; ++i) {
        if (fabs(d_A[i] + d_B[i] - d_C[i]) > 1e-5) {
          fprintf(stderr, "%f %f %f\n", d_A[i], d_B[i], d_C[i]);
          fprintf(stderr, "Result verification failed at element %d!\n", i);
          exit(EXIT_FAILURE);
        }
      }

      printf("Test PASSED\n");
  }

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  /*
  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  */

  printf("Done\n");
  return 0;
}
