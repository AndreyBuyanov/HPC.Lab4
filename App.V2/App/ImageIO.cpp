#include "ImageIO.hpp"

#include <stdexcept>

#include "FreeImage.h"

static bool IsPowerOfTwo(
    const int x)
{
    return x && (!(x & (x - 1)));
}

static FIBITMAP* GenericLoader(
    const char* path,
    const int flag)
{
    FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
    fif = FreeImage_GetFileType(path, 0);
    if (fif == FIF_UNKNOWN) {
        fif = FreeImage_GetFIFFromFilename(path);
    }
    if ((fif != FIF_UNKNOWN)
        && FreeImage_FIFSupportsReading(fif))
    {
        return FreeImage_Load(fif, path, flag);
    }
    return nullptr;
}

static bool GenericWriter(
    FIBITMAP* bitmap,
    const char* path,
    const int flag)
{
    FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
    BOOL success = FALSE;

    if (bitmap) {
        fif = FreeImage_GetFIFFromFilename(path);
        if (fif != FIF_UNKNOWN) {
            auto bpp = FreeImage_GetBPP(bitmap);
            if (FreeImage_FIFSupportsWriting(fif)
                && FreeImage_FIFSupportsExportBPP(fif, bpp))
            {
                success = FreeImage_Save(fif, bitmap, path, flag);
            }
        }
    }
    return (success == TRUE);
}

ImageIO::ImageIO()
{
    FreeImage_Initialise();
}

ImageIO::~ImageIO()
{
    FreeImage_DeInitialise();
}

Image ImageIO::Load(const std::string& path) noexcept(false)
{
    auto inputImage = GenericLoader(path.c_str(), 0);
    if (!inputImage) {
        throw std::runtime_error("");
    }

    const auto imageWidth = FreeImage_GetWidth(inputImage);
    const auto imageHeight = FreeImage_GetHeight(inputImage);
    if (!IsPowerOfTwo(imageWidth) || !IsPowerOfTwo(imageHeight)) {
        throw std::runtime_error("");
    }

    auto temp = inputImage;
    inputImage = FreeImage_ConvertTo8Bits(inputImage);
    if (!inputImage) {
        throw std::runtime_error("");
    }
    FreeImage_Unload(temp);

    Image result { imageWidth, imageHeight };
    for (Image::size_t y = 0; y < result.Height(); y++) {
        auto bits = FreeImage_GetScanLine(inputImage, y);
        if (!bits) {
            throw std::runtime_error("");
        }
        memcpy(&result.Data()[y * result.Width()], bits, result.Width());
    }
    FreeImage_Unload(inputImage);

    return result;
}

bool ImageIO::Save(
    const Image& image,
    const std::string& path) noexcept(false)
{
    auto bitmap = FreeImage_Allocate(image.Width(), image.Height(), 24);
    RGBQUAD color;
    for (int x = 0; x < image.Width(); x++) {
        for (int y = 0; y < image.Height(); y++) {
            color.rgbRed    = image.Data()[y * image.Width() + x];
            color.rgbGreen  = image.Data()[y * image.Width() + x];
            color.rgbBlue   = image.Data()[y * image.Width() + x];
            FreeImage_SetPixelColor(bitmap, x, y, &color);
        }
    }
    auto result = GenericWriter(bitmap, path.c_str(), 0);
    FreeImage_Unload(bitmap);
    return result;
}
