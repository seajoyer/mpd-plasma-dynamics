#include "array2d.hpp"

#include <mpi.h>
#include <cstdio>
#include <new>
#include <utility>

// ---- private helpers -------------------------------------------------------

void Array2D::allocate(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;

    data_ = new (std::nothrow) double*[rows];
    if (!data_) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "FATAL on rank %d: failed to allocate %d row pointers\n",
                rank, rows);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < rows; ++i) {
        data_[i] = new (std::nothrow) double[cols]();   // value-initialised to 0
        if (!data_[i]) {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            fprintf(stderr, "FATAL on rank %d: failed to allocate row %d/%d\n",
                    rank, i, rows);
            for (int j = 0; j < i; ++j) delete[] data_[j];
            delete[] data_;
            data_ = nullptr;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void Array2D::release() noexcept {
    if (data_) {
        for (int i = 0; i < rows_; ++i) delete[] data_[i];
        delete[] data_;
        data_ = nullptr;
    }
    rows_ = cols_ = 0;
}

// ---- public interface -------------------------------------------------------

Array2D::Array2D(int rows, int cols) {
    allocate(rows, cols);
}

Array2D::~Array2D() {
    release();
}

Array2D::Array2D(Array2D&& o) noexcept
    : rows_(o.rows_), cols_(o.cols_), data_(o.data_) {
    o.data_ = nullptr;
    o.rows_ = o.cols_ = 0;
}

Array2D& Array2D::operator=(Array2D&& o) noexcept {
    if (this != &o) {
        release();
        rows_ = o.rows_;
        cols_ = o.cols_;
        data_ = o.data_;
        o.data_ = nullptr;
        o.rows_ = o.cols_ = 0;
    }
    return *this;
}

void Array2D::resize(int rows, int cols) {
    release();
    allocate(rows, cols);
}

void Array2D::fill(double val) {
    for (int i = 0; i < rows_; ++i)
        for (int j = 0; j < cols_; ++j)
            data_[i][j] = val;
}
