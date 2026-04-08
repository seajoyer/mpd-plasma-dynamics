#include "array2d.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>

// ---- private helpers -------------------------------------------------------

void Array2D::Allocate(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;

    // Row-pointer array
    data_ = new (std::nothrow) double*[rows];
    if (!data_) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "FATAL on rank %d: failed to allocate %d row pointers\n", rank,
                rows);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---------------------------------------------------------------
    // SINGLE contiguous data slab  (value-initialised to 0 via '()').
    // ---------------------------------------------------------------
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    auto* block = new (std::nothrow) double[total]();  // zero-init
    if (!block) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "FATAL on rank %d: failed to allocate %zu doubles (%d x %d)\n",
                rank, total, rows, cols);
        delete[] data_;
        data_ = nullptr;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < rows; ++i) {
        data_[i] = block + static_cast<std::size_t>(i) * cols;
    }
}

void Array2D::Release() noexcept {
    if (data_) {
        delete[] data_[0];  // free the single contiguous slab
        delete[] data_;
        data_ = nullptr;
    }
    rows_ = cols_ = 0;
}

// ---- public interface -------------------------------------------------------

Array2D::Array2D(int rows, int cols) { Allocate(rows, cols); }

Array2D::~Array2D() { Release(); }

Array2D::Array2D(Array2D&& o) noexcept : rows_(o.rows_), cols_(o.cols_), data_(o.data_) {
    o.data_ = nullptr;
    o.rows_ = o.cols_ = 0;
}

auto Array2D::operator=(Array2D&& o) noexcept -> Array2D& {
    if (this != &o) {
        Release();
        rows_ = o.rows_;
        cols_ = o.cols_;
        data_ = o.data_;
        o.data_ = nullptr;
        o.rows_ = o.cols_ = 0;
    }
    return *this;
}

void Array2D::Resize(int rows, int cols) {
    Release();
    Allocate(rows, cols);
}

void Array2D::Fill(double val) {
    if (!data_ || rows_ == 0 || cols_ == 0) return;
    // The slab is contiguous → one call instead of rows nested loops.
    std::fill(data_[0], data_[0] + Size(), val);
}
