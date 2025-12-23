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

void AllocateFields(PhysicalFields& fields, int rows, int cols) {
    MemoryAllocation2D(fields.rho, rows, cols);
    MemoryAllocation2D(fields.v_r, rows, cols);
    MemoryAllocation2D(fields.v_phi, rows, cols);
    MemoryAllocation2D(fields.v_z, rows, cols);
    MemoryAllocation2D(fields.e, rows, cols);
    MemoryAllocation2D(fields.p, rows, cols);
    MemoryAllocation2D(fields.P, rows, cols);
    MemoryAllocation2D(fields.H_r, rows, cols);
    MemoryAllocation2D(fields.H_phi, rows, cols);
    MemoryAllocation2D(fields.H_z, rows, cols);
}

void DeallocateFields(PhysicalFields& fields, int rows) {
    MemoryClearing2D(fields.rho, rows);
    MemoryClearing2D(fields.v_r, rows);
    MemoryClearing2D(fields.v_phi, rows);
    MemoryClearing2D(fields.v_z, rows);
    MemoryClearing2D(fields.e, rows);
    MemoryClearing2D(fields.p, rows);
    MemoryClearing2D(fields.P, rows);
    MemoryClearing2D(fields.H_r, rows);
    MemoryClearing2D(fields.H_phi, rows);
    MemoryClearing2D(fields.H_z, rows);
}

void AllocateConservative(ConservativeVars& u, int rows, int cols) {
    MemoryAllocation2D(u.u_1, rows, cols);
    MemoryAllocation2D(u.u_2, rows, cols);
    MemoryAllocation2D(u.u_3, rows, cols);
    MemoryAllocation2D(u.u_4, rows, cols);
    MemoryAllocation2D(u.u_5, rows, cols);
    MemoryAllocation2D(u.u_6, rows, cols);
    MemoryAllocation2D(u.u_7, rows, cols);
    MemoryAllocation2D(u.u_8, rows, cols);
}

void DeallocateConservative(ConservativeVars& u, int rows) {
    MemoryClearing2D(u.u_1, rows);
    MemoryClearing2D(u.u_2, rows);
    MemoryClearing2D(u.u_3, rows);
    MemoryClearing2D(u.u_4, rows);
    MemoryClearing2D(u.u_5, rows);
    MemoryClearing2D(u.u_6, rows);
    MemoryClearing2D(u.u_7, rows);
    MemoryClearing2D(u.u_8, rows);
}

void AllocatePreviousState(PreviousState& prev, int rows, int cols) {
    MemoryAllocation2D(prev.rho_prev, rows, cols);
    MemoryAllocation2D(prev.v_z_prev, rows, cols);
    MemoryAllocation2D(prev.v_r_prev, rows, cols);
    MemoryAllocation2D(prev.v_phi_prev, rows, cols);
    MemoryAllocation2D(prev.H_z_prev, rows, cols);
    MemoryAllocation2D(prev.H_r_prev, rows, cols);
    MemoryAllocation2D(prev.H_phi_prev, rows, cols);
}

void DeallocatePreviousState(PreviousState& prev, int rows) {
    MemoryClearing2D(prev.rho_prev, rows);
    MemoryClearing2D(prev.v_z_prev, rows);
    MemoryClearing2D(prev.v_r_prev, rows);
    MemoryClearing2D(prev.v_phi_prev, rows);
    MemoryClearing2D(prev.H_z_prev, rows);
    MemoryClearing2D(prev.H_r_prev, rows);
    MemoryClearing2D(prev.H_phi_prev, rows);
}
