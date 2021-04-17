#include <iostream>
#include <string>
#include <filesystem>

#include <cuda.h>
#include <cuda_runtime.h>

#include <CLI/CLI.hpp>

#include "Image.hpp"
#include "ImageIO.hpp"

namespace fs = std::filesystem;

void PrintGPUInfo() {
    int deviceCount;
    cudaDeviceProp devProp;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " devices" << std::endl;
    for (int device = 0; device < deviceCount; device++) {
        cudaGetDeviceProperties(&devProp, device);
        std::cout << "Device " << device << std::endl;
        std::cout << "Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
        std::cout << "Name: " << devProp.name << std::endl;
        std::cout << "Total Global Memory: " << devProp.totalGlobalMem << std::endl;
        std::cout << "Shared memory per block: " << devProp.sharedMemPerBlock << std::endl;
        std::cout << "Registers per block: " << devProp.regsPerBlock << std::endl;
        std::cout << "Warp size: " << devProp.warpSize << std::endl;
        std::cout << "Max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "Total constant memory: " << devProp.totalConstMem << std::endl;
    }
}

int main (int argc, char *argv[]){
    
    CLI::App app { "HPC. Laboratory work 4" };

    app.set_help_flag();

    auto help = app.add_flag("-h, --help", "Show this message");
    auto info = app.add_flag("-g, --gpu-info", "Show GPU info");

    std::string inputFilePath;
    app.add_option("-i, --input", inputFilePath,
        "Calculate Gaussian levels")->check(CLI::ExistingFile);

    try {
        app.parse(argc, argv);
        if (*help || argc == 1) {
            throw CLI::CallForHelp();
        }
        if (*info) {
            PrintGPUInfo();
            return EXIT_SUCCESS;
        }
    }
    catch (const CLI::Error& e) {
        return app.exit(e);
    }

    ImageIO io;

    try {
        auto inputImage = io.Load(inputFilePath);
        auto result = inputImage.GetGaussianPyramid(6);
        auto path = fs::path(inputFilePath);
        auto name = path.stem();
        auto ext = path.extension();
        for (int level = 0; level < result.size(); level++) {
            const std::string outputPath =
                name.string() + std::to_string(level + 1) + ext.string();
            if (!io.Save(result[level], outputPath)) {
                std::cout << "" << std::endl;
            }
        }
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
