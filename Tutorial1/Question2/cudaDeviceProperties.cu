#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>    // for std::setw("width") setting (works only for current cout-output)
int getCoresPerSM(int major, int minor) {
    // Defines cores per SM based on architecture generation
    switch (major) {
 case 2: // Fermi
            return (minor == 1) ? 48 : 32;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            if (minor == 1 || minor == 2) return 128;
            if (minor == 0) return 64;
            return 128; // Default fallback for Pascal
        case 7: // Volta (7.0), Turing (7.5)
            return 64;
        case 8: // Ampere (8.0, 8.6, 8.7), Ada Lovelace (8.9)
            if (minor == 0) return 64;
            if (minor == 6 || minor == 9) return 128;
            return 64; // Default fallback for Ampere variants
        case 9: // Hopper (9.0), Blackwell (9.5)
            return 128;
        default:
            return 128; // Standard fallback for future architectures
    }
}
int main() {
    int deviceCount = 0;
    

    int driverVersion = 0;
    int runtimeVersion = 0;

    int clockRate = 0;
    int memoryClockRate = 0;

    int memoryBusWidth = 0;

    // Get the total number of CUDA-enabled devices
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s).\n" << std::endl;

    // Loop through each available device
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        
        

        
        // Populate the property structure for the current device index
        cudaGetDeviceProperties(&prop, i);

        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, i);

        memoryClockRate = prop.memoryClockRate;
        memoryBusWidth = prop.memoryBusWidth;


        std::cout << std::left;
        std::cout << "CUDA Driver Version " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
        std::cout << "CUDA Runtime Version " << runtimeVersion / 1000  << "." << (runtimeVersion % 100) / 10 << std::endl;

        std::cout << "--- Device " << i << ": " << prop.name << " ---" << std::endl;
        std::cout << std::setw(55) << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << std::setw(55) << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << std::setw(55) << "  Streaming Multiprocessors:" << prop.multiProcessorCount << std::endl;
        std::cout << std::setw(55) << "  Cores Per SM: " << getCoresPerSM(prop.major, prop.minor) << std::endl;
        std::cout << std::setw(55) << "  Total Cores: " << prop.multiProcessorCount*getCoresPerSM(prop.major, prop.minor) << std::endl;
        std::cout << std::setw(55) << "  GPU Max Clock Rate: " << clockRate * 1e-6f << " GHz" << std::endl;
        // Memory clock rate should work if CUDA runtime is >= 5.0
        std::cout << std::setw(55) << "  Memory clock rate: " << memoryClockRate * 1e-6f << " GHz" << std::endl;
        std::cout << std::setw(55) << "  Memory Bus Width: " << memoryBusWidth << " bits" << std::endl;
        std::cout << std::setw(55) << "  L2 Cache size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << std::setw(55) << "  Total amount of constant memory: " << prop.totalConstMem << " bytes" << std::endl;
        std::cout << std::setw(55) << "  Total amount of shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << std::setw(55) << "  Total shared memory per multiprocessor (SM): " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
        std::cout << std::setw(55) << "  Total number of registers available per block: " << prop.regsPerBlock << std::endl;
        std::cout << std::setw(55) << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << std::setw(55) << "  Max number of threads per multiprocessor (SM): " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << std::setw(55) << "  Max number of threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << std::setw(55) << "  Max dimension size of a grid: " << "(" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << std::setw(55) << "  Max dimension size of a thread block: " << "(" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        // std::cout << "  Max Threads Per Block:       " << prop.maxThreadsPerBlock << std::endl;
        // std::cout << "  Shared Memory Per Block :     " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        // std::cout << "  Warp Size:                   " << prop.warpSize << std::endl;
        std::cout << std::setw(55) << std::endl;
        std::cout << std::right;
    }

    return 0;
}

