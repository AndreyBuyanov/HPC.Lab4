#pragma once

#include <array>
#include <vector>

class Image {
public:
    using size_t = std::uint32_t;
private:
    using surface_t = std::vector<uint8_t>;

    size_t m_width = 0;
    size_t m_height = 0;
    surface_t m_image;
public:
    Image() = default;
    Image(
        const size_t width,
        const size_t height) :
        m_width(width),
        m_height(height),
        m_image(static_cast<std::size_t>(m_width* m_height)) {}
    size_t Width() const
    {
        return m_width;
    }
    size_t Height() const
    {
        return m_height;
    }
    uint8_t* Data()
    {
        return m_image.data();
    }
    const uint8_t* Data() const
    {
        return m_image.data();
    }

    std::vector<Image> GetGaussianPyramid(
        const std::size_t levels) const;
};
