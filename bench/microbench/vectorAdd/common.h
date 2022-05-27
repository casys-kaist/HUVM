#ifndef __COMMON_H__
#define __COMMON_H__

#include <unistd.h>
#include <stdarg.h>

#define BLOCK_SIZE (1<<19)   // (2 MB / 4 Byte)

#define GPU_FROM 0
#define GPU_TO 1

#define CUDA_ERR_CHECK(x) \
    do { cudaError_t err = x; if (err != cudaSuccess) {          \
        fprintf (stderr, "Error \"%s\" at %s:%d \n",         \
         cudaGetErrorString(err),                            \
        __FILE__, __LINE__); exit(-1);                       \
    }} while (0);

void mark(const char* format, ...)
{
    printf("[Mark] ");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf(" (Start in 1 sec)\n");
    sleep(1);
}

void hold(const char* format, ...)
{
    printf("[Hold] ");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf(" (Press ENTER to continue)");
    while(getchar() != '\n');
}

void enable_peer_access(int gpu_from, int gpu_to)
{
    cudaSetDevice(gpu_from);
    int can_access = -1;
    cudaDeviceCanAccessPeer(&can_access, gpu_from, gpu_to);
    if (can_access) cudaDeviceEnablePeerAccess(gpu_to, 0);
    else printf("Cannot access peer\n");
}

// Init with an index in a block. It can get an weight to add to each index.
__global__ void init(int *x, int w = 0)
{
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        x[i] = i + w;
    }
}

// Init concurrently with fixed value, paired only with check_concurrent. 
// Use for checking throughput.
__global__ void init_concurrent(int **x, int max_i)
{
    // e.g., <<<40,1024>>>
    // threadIdx.x = [0,1023]
    // blockIdx.x = [0,39]
    // blockDim.x = 1024
    // gridDim.x = 40
    unsigned total_idx = (unsigned) threadIdx.x + blockIdx.x * blockDim.x;
    unsigned i = total_idx / BLOCK_SIZE;
    unsigned j = total_idx % BLOCK_SIZE;

    // max_j == BLOCK_SIZE
    if (i < max_i && j < BLOCK_SIZE)
    {
        x[i][j] = 1;
    }
}

// Test both read and write. It adds 1 to single index in a block and revert
// its value back. It can get an weight to pair with init.
__global__ void check(int *x, int w = 0)
{
    // Check single index
    int i = 0;

    bool read_fail = false;
    int back = x[i];

    if (x[i] != i + w) {
        printf("Fail(Read): x[%d] should be %d (was %d).\n", i, i + w, x[i]);
        read_fail = true;
    }

    x[i] += 1;
    // Print write fail only when read didn't fail.
    if ((x[i] != i + w + 1) && !read_fail) 
        printf("Fail(Write): x[%d] should be %d (was %d)\n", i, i + w + 1, x[i]);

    x[i] = back;
}

// Similar with check(), but check all integers in a block.
__global__ void check_all(int *x, int w = 0)
{   
    int diff_r_cnt = 0;
    int diff_w_cnt = 0;

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        bool read_fail = false;
        int back = x[i];

        // It prints every 100,000 index to prevent bunches of printf.
        if (x[i] != i + w) {
            diff_r_cnt++;
            read_fail = true;

            if (i % 100000 == 0) {
                printf("Fail(Read): x[%d] should be %d (was %d).\n", i, i + w, x[i]);
            }
        }

        x[i] += 1;
        // Print write fail only when read didn't fail.
        if (x[i] != i + w + 1)  {
            diff_w_cnt++;

            if ((i % 100000 == 0) && !read_fail) {
                printf("Fail(Write): x[%d] should be %d (was %d)\n", i, i + w + 1, x[i]);
            }
        }

        // Revert back to original value.
        x[i] = back;
    }

    if (diff_r_cnt != 0) {
        printf("Total read failure cnt: %d\n", diff_r_cnt);
    }

    if (diff_w_cnt != 0) {
        printf("Total write failure cnt: %d\n", diff_w_cnt);
    }
}

// Check concurrently. Only paired with init_concurrent. Use for checking 
// throughput of accessing again.
__global__ void check_concurrent(int **x)
{
    unsigned i = (unsigned) threadIdx.x + blockIdx.x * blockDim.x;
    // BUG: Correct if condition
    if(i < 256*16 - 99)
    {
        x[i][0] += 1;
        if (x[i][0] != 0 + 1) printf("False: %d\n", x[i][0]);
    }
}

// Get allocatable total mem size in Mib
size_t get_allocatable_size()
{
    size_t free = 0;
    size_t total = 0;
    CUDA_ERR_CHECK(cudaMemGetInfo(&free, &total));
    return (free - total*0.00055) / (1024*1024);
}

// Get the number of block from size in Mib
int get_num_block(int size)
{
    return size / (BLOCK_SIZE*sizeof(int) / (1024*1024));
}

// Allocate 2D array with num_block * 2 Mib. Need addr of 2D array and num of 
// blocks to allocate, 2 MiB per each block. It can get an weight.
void allocate_block(int ***ptr_to_arr, int num_block, int w = 0)
{
    cudaMallocManaged(ptr_to_arr, num_block*sizeof(int*));
    for (int i = 0; i < num_block; i++)
    {
        cudaMallocManaged(&(*ptr_to_arr)[i], BLOCK_SIZE*sizeof(int));
        init<<<1,1>>>((*ptr_to_arr)[i], w);
    }
}   

// Free 2D array with num_block * 2 Mib
// Need addr of 2D array and num of blocks to free, 2 Mib per each block
void free_all_block(int ***ptr_to_arr, int num_block)
{
    for (int i = 0; i < num_block; i++)
    {
        cudaFree((*ptr_to_arr)[i]);
    }
    cudaFree((*ptr_to_arr));
}

#endif
