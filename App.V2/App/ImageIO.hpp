#pragma once

#include <string>

#include "Image.hpp"

class ImageIO
{
public:
    ImageIO();

    ~ImageIO();

    Image Load(
        const std::string& path) noexcept(false);

    bool Save(
        const Image& image,
        const std::string& path) noexcept(false);
};


