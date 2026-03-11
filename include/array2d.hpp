#pragma once

/// A lightweight 2-D array of doubles stored as a classic double**.
/// Supports the legacy arr[row][col] syntax throughout the numerical
/// kernels, while providing RAII memory management and move semantics.
class Array2D {
    int     rows_{0};
    int     cols_{0};
    double** data_{nullptr};

    void allocate(int rows, int cols);
    void release() noexcept;

public:
    Array2D() = default;
    Array2D(int rows, int cols);
    ~Array2D();

    // Non-copyable, movable
    Array2D(const Array2D&)            = delete;
    Array2D& operator=(const Array2D&) = delete;
    Array2D(Array2D&&) noexcept;
    Array2D& operator=(Array2D&&) noexcept;

    /// Zero-initialise (or reinitialise) to the given dimensions.
    void resize(int rows, int cols);

    /// Fill every element with val (default 0).
    void fill(double val = 0.0);

    // Access – supports arr[row][col] syntax
    double*&       operator[](int i)       { return data_[i]; }
    double* const& operator[](int i) const { return data_[i]; }

    /// Raw pointer – useful when passing to legacy functions.
    double**       raw()       { return data_; }
    const double** raw() const { return const_cast<const double**>(data_); }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
};
