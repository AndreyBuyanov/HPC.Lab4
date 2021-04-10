#include "Image.hpp"

#include "kernel.hpp"

static void ExpandLine(
    const ImageRGB::size_t inputSize,
    const uint8_t* input,
    uint8_t* output)
{
    output[0] = input[0];
    memcpy(&output[1], input, inputSize * sizeof(uint8_t));
    output[inputSize + 1] = input[inputSize - 1];
}

static ImageRGB PrepareImage(
    const ImageRGB& image)
{
    ImageRGB result { image.Width() + 2, image.Height() + 2 };

    std::size_t inputPosition = 0;
    std::size_t outputPosition = 0;
    ExpandLine(
        image.Width(),
        &image.Data()[inputPosition],
        &result.Data()[outputPosition]);
    for (ImageRGB::size_t line = 0; line < image.Height(); line++)
    {
        inputPosition = line * image.Width();
        outputPosition = (line + 1) * result.Width();
        ExpandLine(
            image.Width(),
            &image.Data()[inputPosition],
            &result.Data()[outputPosition]);
    }
    inputPosition = (image.Height() - 1) * image.Width();
    outputPosition = (result.Height() - 1) * result.Width();
    ExpandLine(
        image.Width(),
        &image.Data()[inputPosition],
        &result.Data()[outputPosition]);
    return result;
}

static void CalculateGaussianLevel(
    const ImageRGB& inputImage,
    ImageRGB& outputImage)
{
    kernel(
        inputImage.Width(),
        inputImage.Height(),
        reinterpret_cast<const uint8_t*>(inputImage.Data()),
        outputImage.Width(),
        outputImage.Height(),
        reinterpret_cast<uint8_t*>(outputImage.Data()));
}

static ImageRGB GetGaussianLevel(
    const ImageRGB& image)
{
    ImageRGB input = PrepareImage(image);
    ImageRGB result { image.Width() / 2, image.Height() / 2 };
    CalculateGaussianLevel(input, result);
    return result;
}

std::vector<ImageRGB> ImageRGB::GetGaussianPyramid(
    const std::size_t levels) const
{
    std::vector<ImageRGB> result(levels);
    result[0] = GetGaussianLevel(*this);
    for (std::size_t level = 1; level < levels; level++) {
        result[level] = GetGaussianLevel(result[level - 1]);
    }
    return result;
}
