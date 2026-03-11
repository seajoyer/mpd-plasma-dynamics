#pragma once

#include <mpi.h>
#include <vector>
#include "config.hpp"

/// Wraps MPI initialisation / finalisation and owns the 1-D domain
/// decomposition along the z-axis (l-direction).
///
/// Ghost-cell exchange helpers operate on raw double** arrays so they
/// can be used with both the conservative and physical field arrays
/// without coupling this class to Fields or Array2D.
class MPIManager {
public:
    int rank{0};
    int size{1};

    // ---- domain decomposition (z/l direction) ----
    int L_per_proc{};           ///< base cells per process
    int l_start{};              ///< first global l index owned by this rank
    int l_end{};                ///< last  global l index owned by this rank
    int local_L{};              ///< number of interior (non-ghost) cells
    int local_L_with_ghosts{};  ///< local_L + 2

    /// Initialises MPI and computes domain decomposition.
    MPIManager(int& argc, char**& argv, const SimConfig& cfg);

    /// Calls MPI_Finalize.
    ~MPIManager();

    // Non-copyable, non-movable
    MPIManager(const MPIManager&)            = delete;
    MPIManager& operator=(const MPIManager&) = delete;

    double wtime() const { return MPI_Wtime(); }

    /// Exchange one layer of ghost cells for a single 2-D array.
    /// @param arr     double** of size [local_L_with_ghosts][ncols]
    /// @param ncols   number of columns (M_max + 1)
    /// The four scratch buffers must be pre-allocated to at least ncols doubles.
    void exchange_ghosts(double** arr, int ncols,
                         double* send_left, double* send_right,
                         double* recv_left, double* recv_right) const;
};
