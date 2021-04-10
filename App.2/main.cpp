#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "FreeImage.h"

#include "Image.hpp"

FIBITMAP* GenericLoader(
    const char* lpszPathName,
    int flag)
{
    FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
    fif = FreeImage_GetFileType(lpszPathName, 0);
    if (fif == FIF_UNKNOWN) {
        fif = FreeImage_GetFIFFromFilename(lpszPathName);
    }
    if ((fif != FIF_UNKNOWN)
        && FreeImage_FIFSupportsReading(fif))
    {
        FIBITMAP* dib = FreeImage_Load(fif, lpszPathName, flag);
        return dib;
    }
    return nullptr;
}

bool GenericWriter(
    FIBITMAP* dib,
    const char* lpszPathName,
    int flag)
{
    FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
    BOOL bSuccess = FALSE;

    if (dib) {
        fif = FreeImage_GetFIFFromFilename(lpszPathName);
        if (fif != FIF_UNKNOWN) {
            WORD bpp = FreeImage_GetBPP(dib);
            if (FreeImage_FIFSupportsWriting(fif)
                && FreeImage_FIFSupportsExportBPP(fif, bpp))
            {
                bSuccess = FreeImage_Save(fif, dib, lpszPathName, flag);
            }
        }
    }
    return (bSuccess == TRUE) ? true : false;
}

void FreeImageErrorHandler(
    FREE_IMAGE_FORMAT fif,
    const char* message)
{
    if (fif != FIF_UNKNOWN) {
        std::cout << "Format: " << FreeImage_GetFormatFromFIF(fif) << ". ";
    }
    std::cout << message << std::endl;
}

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
    
    PrintGPUInfo();

    FreeImage_Initialise();

    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    std::string basic_filename = "test";

    FIBITMAP* inputImage = GenericLoader((basic_filename + ".png").c_str(), 0);;

    FIBITMAP* temp = inputImage;
    inputImage = FreeImage_ConvertTo8Bits(inputImage);
    FreeImage_Unload(temp);

    ImageRGB image{ FreeImage_GetWidth(inputImage), FreeImage_GetHeight(inputImage) };
    for (int y = 0; y < image.Height(); y++) {
        BYTE* bits = FreeImage_GetScanLine(inputImage, y);
        memcpy(&image.Data()[y * image.Width()], bits, FreeImage_GetLine(inputImage));
    }

    auto result = image.GetGaussianPyramid(6);

    int level = 1;
    for (auto& r : result) {
        FIBITMAP* bitmap = FreeImage_Allocate(r.Width(), r.Height(), 24);
        RGBQUAD color;
        for (int i = 0; i < r.Width(); i++) {
            for (int j = 0; j < r.Height(); j++) {
                color.rgbRed = r.Data()[i * r.Width() + j];
                color.rgbGreen = r.Data()[i * r.Width() + j];
                color.rgbBlue = r.Data()[i * r.Width() + j];
                FreeImage_SetPixelColor(bitmap, i, j, &color);
            }
        }
        GenericWriter(bitmap, (basic_filename + std::to_string(level) + ".png").c_str(), 0);
        level++;
    }

    FreeImage_DeInitialise();

    return 0;
}
