#pragma once

void init(
    const size_t size,
    uint8_t* data,
    const uint64_t seed);

void kernel(
    const size_t d1,
    const size_t s1,
    const uint8_t* m1,
    const size_t d2,
    const size_t s2,
    const uint8_t* m2,
    uint8_t* m3);
