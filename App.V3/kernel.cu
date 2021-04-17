#pragma warning(push)
#pragma warning(disable:4267)

#include <curand.h>
#include <curand_kernel.h>

#include "kernel.cuh"

__global__ void init_curand(
    curandState* state,
    const uint64_t seed)
{
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void generate(
    curandState* state,
    uint8_t* data)
{
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = static_cast<uint8_t>(curand_uniform(&state[idx]) * 100);
}

void init(
    const size_t size,
    uint8_t* data,
    const uint64_t seed)
{
    const int n_threads = 1024;
    dim3 grid((size + n_threads - 1) / n_threads);
    dim3 block(n_threads);
    curandState* dev_state = nullptr;
    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&dev_state), size * sizeof(curandState)));
    init_curand <<< grid, block >>> (dev_state, seed);
    gpuCheckError(cudaPeekAtLastError());
    gpuCheckError(cudaDeviceSynchronize());
    uint8_t* data_gpu = nullptr;
    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&data_gpu), size * sizeof(uint8_t)));
    generate <<< grid, block >>> (dev_state, data_gpu);
    gpuCheckError(cudaPeekAtLastError());
    gpuCheckError(cudaDeviceSynchronize());
    gpuCheckError(cudaMemcpy(data, data_gpu, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    gpuCheckError(cudaFree(data_gpu));
    gpuCheckError(cudaFree(dev_state));
}

__global__ void kernel_impl(
    const size_t d1,
    const size_t s1,
    const uint8_t* m1,
    const size_t d2,
    const size_t s2,
    const uint8_t* m2,
    uint8_t* m3)
{
    const size_t m3_x = blockIdx.x * 32 + threadIdx.x;
    const size_t m3_y = blockIdx.y * 32 + threadIdx.y;
    const size_t m2_stride_x = d1 / d2;
    const size_t m2_stride_y = s1 / s2;
    const size_t m2_x = m3_x / m2_stride_x;
    const size_t m2_y = m3_y / m2_stride_y;
    const size_t m3_pos = m3_y * s1 + m3_x;
    const size_t m2_pos = m2_y * s2 + m2_x;
    m3[m3_pos] = (m1[m3_pos] > m2[m2_pos]) ? (m1[m3_pos] - m2[m2_pos]) : 0;
}

void kernel(
    const size_t d1,
    const size_t s1,
    const uint8_t* m1,
    const size_t d2,
    const size_t s2,
    const uint8_t* m2,
    uint8_t* m3)
{
    uint8_t* m1_gpu = nullptr;
    uint8_t* m2_gpu = nullptr;
    uint8_t* m3_gpu = nullptr;

    const size_t m1_size = d1 * s1;
    const size_t m2_size = d2 * s2;
    const size_t m3_size = m1_size;

    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&m1_gpu), m1_size * sizeof(uint8_t)));
    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&m2_gpu), m2_size * sizeof(uint8_t)));
    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&m3_gpu), m3_size * sizeof(uint8_t)));

    gpuCheckError(cudaMemcpy(m1_gpu, m1, m1_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuCheckError(cudaMemcpy(m2_gpu, m2, m2_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuCheckError(cudaMemcpy(m3_gpu, m3, m3_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

    dim3 grid(d1 / 32, s1 / 32);
    dim3 block(32, 32);

    kernel_impl <<< grid, block >>> (d1, s1, m1_gpu, d2, s2, m2_gpu, m3_gpu);

    gpuCheckError(cudaMemcpy(m3, m3_gpu, m3_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    gpuCheckError(cudaFree(m1_gpu));
    gpuCheckError(cudaFree(m2_gpu));
    gpuCheckError(cudaFree(m3_gpu));
}

#pragma warning(pop)
