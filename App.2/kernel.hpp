#pragma once

void kernel(
    const std::size_t inputWidth,
    const std::size_t inputHeight,
    const uint8_t* input,
    const std::size_t outputWidth,
    const std::size_t outputHeight,
    uint8_t* output);
