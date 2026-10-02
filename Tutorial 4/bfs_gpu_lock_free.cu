#include <stdio.h>
#include <cuda_runtime.h>


// ------------------------------------------------------------
// BFS kernel
// ------------------------------------------------------------
//
// One thread is assigned to one vertex.
//
// If this vertex belongs to the previous BFS level,
// the thread examines all of its neighbours.
//
// When several threads try to discover the same vertex,
// atomicCAS ensures that only ONE of them gets to claim it.
//
// ------------------------------------------------------------

__global__ void bfsperlevel(const int *srcPtrs,
                            int *level,
                            const int *dst,
                            int curLevel,
                            int *threadDiscovered,
                            int numVertices)
{
    // Global vertex ID handled by this thread.
    int vertexId = blockIdx.x * blockDim.x + threadIdx.x;


    // Every thread initially says:
    // "I did not discover anything."
    if (vertexId < numVertices)
        threadDiscovered[vertexId] = 0;


    // Ignore extra threads if number of threads > number of vertices.
    if (vertexId >= numVertices)
        return;


    // We only process vertices from the previous BFS level.
    //
    // For example:
    // curLevel = 1
    // process vertices whose level is 0.
    //
    // curLevel = 2
    // process vertices whose level is 1.
    //
    if (level[vertexId] != curLevel - 1)
        return;


    // Examine every outgoing edge of this vertex.
    for (int edge = srcPtrs[vertexId];
         edge < srcPtrs[vertexId + 1];
         ++edge)
    {
        int destVertex = dst[edge];


        // ----------------------------------------------------
        // Try to claim destVertex.
        //
        // Initially, undiscovered vertices have level = 5.
        //
        // We want to change:
        //
        //       5  --->  curLevel
        //
        // Only ONE thread should be allowed to do this.
        //
        // atomicCAS(address, oldValue, newValue)
        //
        // means:
        //
        // if (*address == oldValue)
        //     *address = newValue;
        //
        // and the whole operation happens atomically.
        // ----------------------------------------------------

        int oldLevel = atomicCAS(
            &level[destVertex],
            5,
            curLevel
        );


        // If oldLevel was 5, THIS thread successfully
        // discovered the vertex.
        //
        // If another thread had already changed it,
        // oldLevel will not be 5.
        //
        if (oldLevel == 5)
        {
            threadDiscovered[vertexId] = 1;
        }
    }
}


// ------------------------------------------------------------
// Print an integer array.
// ------------------------------------------------------------

void printArray(const int *arr, int size)
{
    printf("\nArray: ");

    for (int i = 0; i < size; ++i)
        printf("%d, ", arr[i]);

    printf("\n");
}


// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main()
{
    // --------------------------------------------------------
    // Same graph as your CPU/GPU programs.
    //
    // CSR representation.
    // --------------------------------------------------------

    int h_srcPtrs[] = {
        0, 2, 4, 7, 9, 11, 12, 13, 15, 15
    };


    int h_dst[] = {
        1, 2,
        3, 4,
        5, 6, 7,
        4, 8,
        5, 8,
        6,
        8,
        0, 6
    };


    // 0 = source
    // 5 = undiscovered
    int h_level[] = {
        0, 5, 5, 5, 5, 5, 5, 5, 5
    };


    int numVertices = sizeof(h_level) / sizeof(int);
    int numEdges    = sizeof(h_dst)   / sizeof(int);


    // --------------------------------------------------------
    // Device pointers.
    // --------------------------------------------------------

    int *d_srcPtrs;
    int *d_dst;
    int *d_level;
    int *d_threadDiscovered;


    // --------------------------------------------------------
    // Allocate GPU memory.
    // --------------------------------------------------------

    cudaMalloc(&d_srcPtrs,
               (numVertices + 1) * sizeof(int));

    cudaMalloc(&d_dst,
               numEdges * sizeof(int));

    cudaMalloc(&d_level,
               numVertices * sizeof(int));

    cudaMalloc(&d_threadDiscovered,
               numVertices * sizeof(int));


    // --------------------------------------------------------
    // Copy graph to GPU.
    // --------------------------------------------------------

    cudaMemcpy(
        d_srcPtrs,
        h_srcPtrs,
        (numVertices + 1) * sizeof(int),
        cudaMemcpyHostToDevice
    );


    cudaMemcpy(
        d_dst,
        h_dst,
        numEdges * sizeof(int),
        cudaMemcpyHostToDevice
    );


    cudaMemcpy(
        d_level,
        h_level,
        numVertices * sizeof(int),
        cudaMemcpyHostToDevice
    );


    // --------------------------------------------------------
    // BFS variables.
    // --------------------------------------------------------

    int curLevel = 0;

    // At least the source vertex exists,
    // so we need to perform the first iteration.
    int newVertexDiscovered = 1;

    int iterationCount = 0;


    printf("Initial level array:");
    printArray(h_level, numVertices);


    // --------------------------------------------------------
    // BFS loop.
    // --------------------------------------------------------

    do
    {
        if (newVertexDiscovered == 1)
        {
            iterationCount++;

            // Assume no new vertex is discovered.
            // GPU threads will set their individual flags.
            newVertexDiscovered = 0;

            // Move to the next BFS level.
            curLevel++;


            printf("\nIteration %d...\n", iterationCount);


            // ------------------------------------------------
            // Launch one thread per vertex.
            // ------------------------------------------------

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


            // Wait for kernel to finish.
            cudaDeviceSynchronize();


            // ------------------------------------------------
            // Copy per-thread discovery flags to CPU.
            // ------------------------------------------------

            int h_threadDiscovered[numVertices];


            cudaMemcpy(
                h_threadDiscovered,
                d_threadDiscovered,
                numVertices * sizeof(int),
                cudaMemcpyDeviceToHost
            );


            // ------------------------------------------------
            // Did ANY thread discover a new vertex?
            // ------------------------------------------------

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


    // --------------------------------------------------------
    // Copy final levels back to CPU.
    // --------------------------------------------------------

    cudaMemcpy(
        h_level,
        d_level,
        numVertices * sizeof(int),
        cudaMemcpyDeviceToHost
    );


    printf("\nAfter BFS stops:");
    printArray(h_level, numVertices);


    // --------------------------------------------------------
    // Free GPU memory.
    // --------------------------------------------------------

    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_level);
    cudaFree(d_threadDiscovered);


    return 0;
}
