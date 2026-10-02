#include <stdio.h>
#include <cuda_runtime.h>

#define N 4
#define TILE 2
#define NUM_STREAMS 4

// Each CUDA block computes one TILE x TILE portion of C.
// Here TILE = 2, so each block has 2 x 2 = 4 threads.
__global__ void tiledMatMul(int *A, int *B, int *C, int row)
{
    // threadIdx.x selects the column within the tile.
    // threadIdx.y selects the row within the tile.
    int col = blockIdx.x * TILE + threadIdx.x;

    // 'row' is the particular row of C being calculated
    // by this stream.
    
    // Shared memory stores small tiles of A and B.
    // Threads in the block cooperate to load these tiles.
    __shared__ int tileA[TILE][TILE];
    __shared__ int tileB[TILE][TILE];

    int sum = 0;

    // N/TILE = 4/2 = 2.
    // Therefore, two pairs of tiles are needed to calculate
    // one row of C.
    for (int t = 0; t < N / TILE; t++)
    {
        /*
           Load a tile of A into shared memory.

           Since we are calculating only one row,
           threadIdx.y is always 0 in this simple example.
        */
        tileA[threadIdx.y][threadIdx.x] =
            A[row * N + t * TILE + threadIdx.x];

        /*
           Load a tile of B into shared memory.

           Each thread loads one element of B.
        */
        tileB[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];

        // Make sure ALL threads have finished loading
        // the tiles before anyone starts using them.
        __syncthreads();

        // Multiply the two tiles.
        for (int k = 0; k < TILE; k++)
        {
            sum += tileA[threadIdx.y][k] *
                   tileB[k][threadIdx.x];
        }

        // Make sure all threads have finished using the
        // current tiles before loading the next tiles.
        __syncthreads();
    }

    // Store the final value calculated by this thread.
    C[row * N + col] = sum;
}

int main()
{
    /*
       Small example matrices.

       A is a normal 4 x 4 matrix.

       B is the identity matrix, so the expected result
       C = A. This makes checking the output easy.
    */
    int A[N][N] = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
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

    /*
       We use one stream for each row.

       Stream 0 -> calculates row 0
       Stream 1 -> calculates row 1
       Stream 2 -> calculates row 2
       Stream 3 -> calculates row 3
    */
    cudaStream_t streams[NUM_STREAMS];

    // Each stream gets its own device memory.
    // This avoids different streams overwriting each other's data.
    int *d_A[NUM_STREAMS];
    int *d_B[NUM_STREAMS];
    int *d_C[NUM_STREAMS];

    /*
       Create four streams and allocate device memory
       for each stream.
    */
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);

        cudaMalloc(&d_A[i], N * sizeof(int));
        cudaMalloc(&d_B[i], N * N * sizeof(int));
        cudaMalloc(&d_C[i], N * sizeof(int));
    }

    /*
       Process the four rows.

       Each iteration creates the following sequence
       inside one stream:

          Copy A row  -> Copy B -> Kernel -> Copy C row

       Different streams can execute their sequences
       concurrently when the GPU supports it.
    */
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        /*
           Copy only ONE row of A.

           Stream i calculates row i, so it only needs
           A[i][0 ... N-1].
        */
        cudaMemcpyAsync(
            d_A[i],
            A[i],
            N * sizeof(int),
            cudaMemcpyHostToDevice,
            streams[i]
        );

        /*
           B is needed completely because every output
           element uses an entire row of A with a column of B.
        */
        cudaMemcpyAsync(
            d_B[i],
            B,
            N * N * sizeof(int),
            cudaMemcpyHostToDevice,
            streams[i]
        );

        /*
           One block is enough because we calculate only
           one row.

           2 x 2 threads are used because TILE = 2.
        */
        dim3 block(TILE, TILE);

        /*
           There are N/TILE = 2 tiles along the columns.

           So the grid contains 2 blocks in the x direction
           and only 1 block in the y direction.
        */
        dim3 grid(N / TILE, 1);

        /*
           Launch the tiled matrix multiplication kernel
           in stream i.

           'i' tells the kernel which row of C to calculate.
        */
        tiledMatMul<<<grid, block, 0, streams[i]>>>(
            d_A[i],
            d_B[i],
            d_C[i],
            i
        );

        /*
           Copy the calculated row from device to host.

           The copy is asynchronous and belongs to the
           same stream, so it happens AFTER the kernel
           finishes in that stream.
        */
        cudaMemcpyAsync(
            C[i],
            d_C[i],
            N * sizeof(int),
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    /*
       Wait until ALL four streams have completed.

       Without this, the CPU could print C before the
       asynchronous operations have finished.
    */
    cudaDeviceSynchronize();

    // Print the final matrix.
    printf("Result matrix C:\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%d ", C[i][j]);

        printf("\n");
    }

    /*
       Free the device memory and destroy the streams.
    */
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);

        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
