#pragma once

#include <cstddef>

/// A lightweight 2-D array of doubles stored as a classic double**.
/// Supports the legacy arr[row][col] syntax throughout the numerical
/// kernels, while providing RAII memory management and move semantics.
///
/// Performance note: the entire data block is allocated as ONE contiguous
/// slab (rows * cols doubles).  Row-pointers fan out into that slab so that
///   - stencil accesses arr[l±1][m] hit adjacent cache lines instead of
///     scattered heap fragments;
///   - the inner loop over m is a simple stride-1 walk → auto-vectorised;
///   - MPI row packs reduce to memcpy(dst, arr[row], cols*8);
///   - MPI column packs walk arr[l][col] with a known stride of cols.
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

    /// Pointer to the start of the contiguous data slab.
    /// Enables bulk operations: memset, std::fill, MPI send of entire array.
    double*       flat()       { return data_ ? data_[0] : nullptr; }
    const double* flat() const { return data_ ? data_[0] : nullptr; }

    /// Total number of elements (rows * cols).
    std::size_t size() const {
        return static_cast<std::size_t>(rows_) * cols_;
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
};
