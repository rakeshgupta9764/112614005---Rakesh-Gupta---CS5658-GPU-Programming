#include <stdio.h>
#include <cuda_runtime.h>

#define N 4
#define TILE 2

__global__ void tiledMatMul(int *A, int *B, int *C)
{
    // Row and column of C calculated by this thread
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ int tileA[TILE][TILE];
    __shared__ int tileB[TILE][TILE];

    int sum = 0;

    // There are N/TILE tiles
    for (int t = 0; t < N / TILE; t++)
    {
        // Load one element into each shared-memory tile
        tileA[threadIdx.y][threadIdx.x] =
            A[row * N + t * TILE + threadIdx.x];

        tileB[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];

        // Wait until the entire tile is loaded
        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE; k++)
        {
            sum += tileA[threadIdx.y][k] *
                   tileB[k][threadIdx.x];
        }

        // Wait before loading the next tiles
        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main()
{
    int A[N][N] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    int B[N][N] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };

    int C[N][N];

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));

    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    tiledMatMul<<<grid, block>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result matrix C:\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%d ", C[i][j]);

        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
