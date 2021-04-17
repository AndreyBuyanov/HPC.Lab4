#include <random>
#include <algorithm>
#include <execution>

#include "Matrix.hpp"
#include "kernel.hpp"

void Matrix::Init(
    const std::uint64_t seed)
{
    constexpr size_t numChunks = 64;
    const size_t chunkSize = m_matrix.size() / numChunks;
    for (size_t chunk = 0; chunk < numChunks; chunk++) {
        uint8_t* pos = &m_matrix.data()[chunk * chunkSize];
        init(chunkSize, pos, seed);
    }
}

Matrix Matrix::CalculateContrast(
    const Matrix& m1,
    const Matrix& m2)
{
    Matrix result(m1.Rows(), m1.Cols());
    kernel(
        m1.Rows(), m1.Cols(), m1.Data(),
        m2.Rows(), m2.Cols(), m2.Data(),
        result.Data());
    return result;
}
