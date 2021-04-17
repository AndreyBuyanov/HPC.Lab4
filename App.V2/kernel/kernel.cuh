#pragma once

#pragma warning(push)
#pragma warning(disable:26812)

#include <cuda_runtime.h>
#include <iostream>

inline void gpuAssert(
    const cudaError_t code,
    const char* file,
    const int line,
    const bool abort)
{
    if (code != cudaSuccess) {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) {
            exit(code);
        }
    }
}

inline void gpuCheckError(
    const cudaError_t code,
    const bool abort = true)
{
    gpuAssert(code, __FILE__, __LINE__, abort);
}

#pragma warning(pop)
