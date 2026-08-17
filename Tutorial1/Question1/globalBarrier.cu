#include <stdio.h>
#include <cuda_runtime.h>

__device__ int counter = 0;

__global__ void globalBarrierKernel() {

    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1
    // printf("Thread %d: before barrier\n", tid);
    if (threadIdx.x == 0){
        printf("Block %d: BEFORE barrier\n", blockIdx.x);
    }
        

    // Synchronize threads within each block
    __syncthreads();

    // The first thread form each block announces that its block has arrived to (its block-level) barrier
    if (threadIdx.x == 0) {
        atomicAdd(&counter, 1);
        printf("Block %d: ARRIVED, counter = %d\n", blockIdx.x, counter);
    }

    // Global barrier: each thread has to wait until all the blocks (all the threads) have arrived, i.e., wait until counter becomes equal to #blocks
    while (counter < gridDim.x) {
        ;   // infinite loop for waiting
    }

    // Phase 2
    // printf("Thread %d: after barrier\n", tid);
    if (threadIdx.x == 0) {
        printf("Block %d: PASSED barrier, counter = %d\n",
            blockIdx.x, counter);
    }
    

}

int main() {

    int blocks = 4;
    int threadsPerBlock = 64;

    globalBarrierKernel<<<blocks, threadsPerBlock>>>();

    cudaDeviceSynchronize();


    return 0;
}
