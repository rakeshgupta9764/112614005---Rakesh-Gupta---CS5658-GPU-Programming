// tensor_core_matmul_64x64.cu
//
// 64x64 matrix multiplication using NVIDIA Tensor Cores.
// Each matrix is divided into 16x16 tiles.
//
// 64/16 = 4 tiles in each dimension, so C has 4 x 4 = 16 output tiles.
// One CUDA block contains one warp (32 threads) and computes one 16x16 C tile.

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;
using namespace wmma;

const int M = 64;
const int N = 64;
const int K = 64;

// WMMA operation used in this program: 16x16x16.
const int TILE = 16;
const int NUM_TILES = 4;       // 64/16
const int WARP_SIZE = 32;

// ------------------------------------------------------------
// Create simple example matrices on the CPU.
//
// A[i][j] = i + 1     -> every row has one constant value
// B[i][j] = j + 1     -> every column has one constant value
//
// Therefore:
// C[i][j] = 64 * (i+1) * (j+1)
// which makes the result easy to check.
// ------------------------------------------------------------
void initializeMatrices(half *A, half *B)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
            A[i * K + j] = __float2half((float)(i + 1));
    }

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
            B[i * N + j] = __float2half((float)(j + 1));
    }
}

// ------------------------------------------------------------
// One block = one warp = one 16x16 output tile of C.
// Grid = 4x4 blocks, so all 16 C tiles are computed.
// ------------------------------------------------------------
__global__ void tensorCoreMatMul(const half *A, const half *B, float *C)
{
    // Identify the 16x16 tile of C computed by this block.
    int tileRow = blockIdx.y;       // 0..3
    int tileCol = blockIdx.x;       // 0..3

    // Top-left element of this tile in the full matrix.
    int rowStart = tileRow * TILE;
    int colStart = tileCol * TILE;

    // WMMA fragments hold pieces of matrices distributed among
    // the 32 threads of the warp.
    fragment<matrix_a, TILE, TILE, TILE, half, row_major> aFrag;
    fragment<matrix_b, TILE, TILE, TILE, half, row_major> bFrag;
    fragment<accumulator, TILE, TILE, TILE, float> cFrag;

    // Start the output tile at zero.
    fill_fragment(cFrag, 0.0f);

    // K = 64, but one WMMA multiply uses K = 16.
    // Therefore four 16-wide pieces must be accumulated.
    for (int kTile = 0; kTile < NUM_TILES; kTile++)
    {
        // 16x16 tile A(tileRow, kTile)
        const half *tileA = A + rowStart * K + kTile * TILE;

        // 16x16 tile B(kTile, tileCol)
        const half *tileB = B + kTile * TILE * N + colStart;

        // The whole warp cooperatively loads the tiles.
        load_matrix_sync(aFrag, tileA, K);
        load_matrix_sync(bFrag, tileB, N);

        // Tensor Core matrix multiply-accumulate:
        // cFrag = aFrag * bFrag + cFrag
        mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    // Top-left element of the 16x16 C tile.
    float *tileC = C + rowStart * N + colStart;

    // Store the completed tile back to global memory.
    store_matrix_sync(tileC, cFrag, N, mem_row_major);
}

int main()
{
    // Input matrices are FP16 (half precision).
    half hA[M * K];
    half hB[K * N];

    // Output is FP32.
    float hC[M * N];

    half *dA;
    half *dB;
    float *dC;

    // Allocate GPU memory.
    cudaMalloc(&dA, M * K * sizeof(half));
    cudaMalloc(&dB, K * N * sizeof(half));
    cudaMalloc(&dC, M * N * sizeof(float));

    // Prepare the simple example input.
    initializeMatrices(hA, hB);

    // Copy A and B from CPU memory to GPU memory.
    cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // --------------------------------------------------------
    // Launch the kernel.
    //
    // 4 x 4 blocks  -> 16 output tiles
    // 32 threads    -> exactly one warp per block
    // --------------------------------------------------------
    dim3 grid(NUM_TILES, NUM_TILES);
    dim3 block(WARP_SIZE);

    tensorCoreMatMul<<<grid, block>>>(dA, dB, dC);

    // Wait until the GPU finishes the kernel.
    cudaDeviceSynchronize();

    // Copy the result C back to the CPU.
    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print only the top-left 4x4 part, to keep the output readable.
    printf("Top-left 4x4 part of C:\n\n");

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            printf("%8.1f ", hC[i * N + j]);
        printf("\n");
    }

    // Check the complete 64x64 result.
    float maxError = 0.0f;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float expected = (float)K * (i + 1) * (j + 1);
            float error = fabsf(hC[i * N + j] - expected);

            if (error > maxError)
                maxError = error;
        }
    }

    printf("\nMaximum error = %f\n", maxError);

    if (maxError < 0.001f)
        printf("Result check: PASS\n");
    else
        printf("Result check: FAIL\n");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}

// Output:
//
// Top-left 4x4 part of C:
//
//     64.0    128.0    192.0    256.0
//    128.0    256.0    384.0    512.0
//    192.0    384.0    576.0    768.0
//    256.0    512.0    768.0   1024.0
//
// Maximum error = 0.000000
// Result check: PASS
//
// TO be compiled on a Tensor-Core-capable GPU. WMMA/Tensor Core matrix operations
// require compute capability 7.0 or higher. For example:
//
//     nvcc -arch=sm_70 tensor_core_matmul_64x64.cu -o tensor_core_matmul
//
// For a newer GPU, we can replace sm_70 with its compute capability, e.g. sm_80.
