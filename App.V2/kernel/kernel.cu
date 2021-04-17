#include <stdint.h>

#include "kernel.cuh"

__global__ void kernel_impl(
    const uint8_t* input,
    const size_t inputWidth,
    uint8_t* output,
    const size_t outputWidth)
{
    int outputX = blockIdx.x * 32 + threadIdx.x;
    int outputY = blockIdx.y * 32 + threadIdx.y;
    int inputXBasic = outputX * 2;
    int inputYBasic = outputY * 2;
    float result = 0.0f;
    for (int x = 0; x < 4; x++) {
        for (int y = 0; y < 4; y++) {
            int inputX = inputXBasic + x;
            int inputY = inputYBasic + y;
            int inputIndex = inputY * inputWidth + inputX;
            result += input[inputIndex];
        }
    }
    output[outputY * outputWidth + outputX] = static_cast<uint8_t>(result / 16.0f);
}

void kernel(
    const std::size_t inputWidth,
    const std::size_t inputHeight,
    const uint8_t* input,
    const std::size_t outputWidth,
    const std::size_t outputHeight,
    uint8_t* output)
{
    dim3 grid(outputWidth / 32, outputHeight / 32);
    dim3 block(32, 32);

    const std::size_t inputSize = inputWidth * inputHeight;
    const std::size_t outputSize = outputWidth * outputHeight;

    uint8_t* inputGpu;
    uint8_t* outputGpu;

    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&inputGpu), inputSize * sizeof(uint8_t)));
    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&outputGpu), outputSize * sizeof(uint8_t)));

    gpuCheckError(cudaMemcpy(inputGpu, input, inputSize * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuCheckError(cudaMemcpy(outputGpu, output, outputSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

    kernel_impl <<< grid, block >>> (inputGpu, inputWidth, outputGpu, outputWidth);

    gpuCheckError(cudaMemcpy(output, outputGpu, outputSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    gpuCheckError(cudaFree(inputGpu));
    gpuCheckError(cudaFree(outputGpu));
}
