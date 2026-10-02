#include <stdio.h>
#include <cuda_runtime.h>


__global__ void bfsperlevel(const int *srcPtrs,
                            int *level,
                            const int *dst,
                            int curLevel,
                            int *threadDiscovered,
                            int numVertices)
{
    int vertexId = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread starts with "no new vertex discovered".
    threadDiscovered[vertexId] = 0;

    if (vertexId < numVertices &&
        level[vertexId] == curLevel - 1)
    {
        // Consider every neighbour of this vertex.
        for (int edge = srcPtrs[vertexId];
             edge < srcPtrs[vertexId + 1];
             ++edge)
        {
            int destVertex = dst[edge];

            if (level[destVertex] > curLevel)
            {
                level[destVertex] = curLevel;

                // This thread discovered a new vertex.
                threadDiscovered[vertexId] = 1;
            }
        }
    }
}


void printArray(const int *arr, int size)
{
    printf("\nArray: ");

    for (int i = 0; i < size; ++i)
        printf("%d, ", arr[i]);

    printf("\n");
}


int main()
{
    // Same graph as the CPU program.
    int h_srcPtrs[] = {0,2,4,7,9,11,12,13,15,15};

    int h_dst[] = {
        1,2,3,4,5,6,7,4,8,5,8,6,8,0,6
    };

    int h_level[] = {
        0,5,5,5,5,5,5,5,5
    };


    int numVertices = sizeof(h_level) / sizeof(int);
    int numEdges = sizeof(h_dst) / sizeof(int);


    // Device arrays.
    int *d_srcPtrs;
    int *d_dst;
    int *d_level;
    int *d_threadDiscovered;


    cudaMalloc(&d_srcPtrs,
               (numVertices + 1) * sizeof(int));

    cudaMalloc(&d_dst,
               numEdges * sizeof(int));

    cudaMalloc(&d_level,
               numVertices * sizeof(int));

    cudaMalloc(&d_threadDiscovered,
               numVertices * sizeof(int));


    // Copy input data to GPU.
    cudaMemcpy(d_srcPtrs,
               h_srcPtrs,
               (numVertices + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_dst,
               h_dst,
               numEdges * sizeof(int),
               cudaMemcpyHostToDevice);


    // Source vertex is vertex 0.
    h_level[0] = 0;

    cudaMemcpy(d_level,
               h_level,
               numVertices * sizeof(int),
               cudaMemcpyHostToDevice);


    int curLevel = 0;
    int newVertexDiscovered = 1;
    int iterationCount = 0;


    printf("Initial level array:");
    printArray(h_level, numVertices);


    // Same level-by-level logic as the CPU version.
    do
    {
        if (newVertexDiscovered == 1)
        {
            iterationCount++;

            newVertexDiscovered = 0;
            curLevel++;

            printf("\nIteration %d...\n", iterationCount);


            int threadsPerBlock = 256;

            int blocks =
                (numVertices + threadsPerBlock - 1)
                / threadsPerBlock;


            bfsperlevel<<<blocks, threadsPerBlock>>>(
                d_srcPtrs,
                d_level,
                d_dst,
                curLevel,
                d_threadDiscovered,
                numVertices
            );


            // Wait for all GPU threads to finish.
            cudaDeviceSynchronize();


            // Bring the per-thread flags back to CPU.
            int h_threadDiscovered[numVertices];

            cudaMemcpy(h_threadDiscovered,
                       d_threadDiscovered,
                       numVertices * sizeof(int),
                       cudaMemcpyDeviceToHost);


            // Equivalent to:
            //
            // if ANY thread discovered a new vertex,
            // newVertexDiscovered = 1.
            //
            for (int i = 0; i < numVertices; ++i)
            {
                if (h_threadDiscovered[i] == 1)
                {
                    newVertexDiscovered = 1;
                    break;
                }
            }
        }
        else
        {
            printf("\nBFS finished.\n");
        }

    } while (newVertexDiscovered == 1);


    // Copy final level array back to CPU.
    cudaMemcpy(h_level,
               d_level,
               numVertices * sizeof(int),
               cudaMemcpyDeviceToHost);


    printf("\nAfter BFS stops:");
    printArray(h_level, numVertices);


    // Free GPU memory.
    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_level);
    cudaFree(d_threadDiscovered);


    return 0;
}
