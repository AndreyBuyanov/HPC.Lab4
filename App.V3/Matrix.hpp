#pragma once

#include <vector>

#pragma warning(push)
#pragma warning(disable:26451)

class Matrix {
public:
    using value_type = std::uint8_t;
    using size_type = std::uint32_t;
private:
    std::vector<value_type> m_matrix;
    size_type m_rows = 0;
    size_type m_cols = 0;
public:
    Matrix() = default;
    Matrix(
        const size_type rows,
        const size_type cols) :
        m_rows(rows),
        m_cols(cols),
        m_matrix(static_cast<std::size_t>(m_rows* m_cols)) {}
    void Init(
        const std::uint64_t seed);
    size_type Rows() const
    {
        return m_rows;
    }
    size_type Cols() const
    {
        return m_cols;
    }
    value_type* Data()
    {
        return m_matrix.data();
    }
    const value_type* Data() const
    {
        return m_matrix.data();
    }
    static Matrix CalculateContrast(
        const Matrix& m1,
        const Matrix& m2);
};

#pragma warning(pop)
