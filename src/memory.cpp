#include "memory.hpp"
#include <new>
#include <cstdio>
#include <mpi.h>

void MemoryAllocation2D(double** &array, int rows, int columns) {
    array = new (std::nothrow) double*[rows];
    if (!array) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "FATAL ERROR on rank %d: Failed to allocate %d row pointers\n", rank, rows);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (int i = 0; i < rows; i++) {
        array[i] = new (std::nothrow) double[columns];
        if (!array[i]) {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            fprintf(stderr, "FATAL ERROR on rank %d: Failed to allocate row %d/%d (columns=%d)\n", 
                    rank, i, rows, columns);
            // Clean up previously allocated rows
            for (int j = 0; j < i; j++) {
                delete[] array[j];
            }
            delete[] array;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Initialize to zero
        for (int j = 0; j < columns; j++) {
            array[i][j] = 0.0;
        }
    }
}

void MemoryClearing2D(double** &array, int rows) {
    for (int i = 0; i < rows; i++) {
        delete[] array[i];
    }
    delete[] array;
}
