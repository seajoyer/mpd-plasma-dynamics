#pragma once

#include <mpi.h>
#include <vector>
#include "config.hpp"

/// Wraps MPI initialisation / finalisation and owns a 2-D Cartesian domain
/// decomposition along both the z-axis (l-direction) and r-axis (m-direction).
///
/// Indexing convention (mirrors Fields/Grid):
///   Interior cells : l ∈ [1..local_L],  m ∈ [1..local_M]
///   Ghost cells    : l = 0, l = local_L+1,  m = 0, m = local_M+1
///   Bottom rank (coords[1]==0) : m_local=1 == global m=0  (inner wall cell)
///   Top    rank (coords[1]==dims[1]-1) : m_local=local_M == global m=M_max
///
/// Ghost exchange operates in all four directions in one non-blocking phase.
class MPIManager {
public:
    int rank{0};
    int size{1};

    /// Cartesian communicator (created with reorder=0 so rank == WORLD rank).
    MPI_Comm cart_comm{MPI_COMM_NULL};
    int dims[2]{0, 0};    ///< [0] = #procs in l-dir,  [1] = #procs in m-dir
    int coords[2]{0, 0};  ///< this rank's position in the 2-D grid

    // ---- L decomposition (dimension 0, z-axis) ----
    int L_per_proc{};
    int l_start{}, l_end{};
    int local_L{}, local_L_with_ghosts{};

    // ---- M decomposition (dimension 1, r-axis) ----
    int M_per_proc{};
    int m_start{}, m_end{};
    int local_M{}, local_M_with_ghosts{};

    // ---- Cartesian neighbours ----
    int nbr_l_lo{MPI_PROC_NULL}, nbr_l_hi{MPI_PROC_NULL};
    int nbr_m_lo{MPI_PROC_NULL}, nbr_m_hi{MPI_PROC_NULL};

    // ---- boundary predicates ----
    bool is_l_lo_boundary() const { return coords[0] == 0; }
    bool is_l_hi_boundary() const { return coords[0] == dims[0] - 1; }
    bool is_m_lo_boundary() const { return coords[1] == 0; }
    bool is_m_hi_boundary() const { return coords[1] == dims[1] - 1; }

    /// Initialises MPI and computes 2-D domain decomposition.
    MPIManager(int& argc, char**& argv, const SimConfig& cfg);

    /// Frees the Cartesian communicator and calls MPI_Finalize.
    ~MPIManager();

    // Non-copyable, non-movable
    MPIManager(const MPIManager&)            = delete;
    MPIManager& operator=(const MPIManager&) = delete;

    double wtime() const { return MPI_Wtime(); }

    /// Exchange one layer of ghost cells in all four Cartesian directions for
    /// a single 2-D array.
    ///
    /// @param arr     double** of size [local_L_with_ghosts][local_M_with_ghosts]
    ///
    /// Row buffers (l-direction, contiguous sends):
    ///   row_sl / row_sr : send buffers, size >= local_M_with_ghosts
    ///   row_rl / row_rr : recv buffers, size >= local_M_with_ghosts
    ///
    /// Column buffers (m-direction, packed sends):
    ///   col_sl / col_sr : send buffers, size >= local_L_with_ghosts
    ///   col_rl / col_rr : recv buffers, size >= local_L_with_ghosts
    ///
    /// All four directions are issued non-blocking and waited on in a single
    /// MPI_Waitall.  Corner ghosts are never read by the 5-point stencil so
    /// no special treatment is required.
    void exchange_ghosts(double** arr,
                         double* row_sl, double* row_sr,
                         double* row_rl, double* row_rr,
                         double* col_sl, double* col_sr,
                         double* col_rl, double* col_rr) const;
};
