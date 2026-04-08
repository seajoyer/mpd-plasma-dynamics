#pragma once

#include <cstddef>

/// A lightweight 2-D array of doubles stored as a classic double**.
class Array2D {
    int rows_{0};
    int cols_{0};
    double** data_{nullptr};

    void Allocate(int rows, int cols);
    void Release() noexcept;

   public:
    Array2D() = default;
    Array2D(int rows, int cols);
    ~Array2D();

    // Non-copyable, movable
    Array2D(const Array2D&) = delete;
    auto operator=(const Array2D&) -> Array2D& = delete;
    Array2D(Array2D&&) noexcept;
    auto operator=(Array2D&&) noexcept -> Array2D&;

    /// Zero-initialise (or reinitialise) to the given dimensions.
    void Resize(int rows, int cols);

    /// Fill every element with val (default 0).
    void Fill(double val = 0.0);

    // Access – supports arr[row][col] syntax
    auto operator[](int i) -> double*& { return data_[i]; }
    auto operator[](int i) const -> double* const& { return data_[i]; }

    /// Raw pointer – useful when passing to legacy functions.
    auto Raw() -> double** { return data_; }
    [[nodiscard]] auto Raw() const -> const double** {
        return const_cast<const double**>(data_);
    }

    /// Pointer to the start of the contiguous data slab.
    /// Enables bulk operations: memset, std::fill, MPI send of entire array.
    auto Flat() -> double* { return data_ ? data_[0] : nullptr; }
    [[nodiscard]] auto Flat() const -> const double* {
        return data_ ? data_[0] : nullptr;
    }

    /// Total number of elements (rows * cols).
    [[nodiscard]] auto Size() const -> std::size_t {
        return static_cast<std::size_t>(rows_) * cols_;
    }

    [[nodiscard]] auto Rows() const -> int { return rows_; }
    [[nodiscard]] auto Cols() const -> int { return cols_; }
};
