#pragma once

#include <mpi.h>
#include <vector>
#include "config.hpp"

/// Wraps MPI initialisation / finalisation and owns a 2-D Cartesian domain
/// decomposition along both the z-axis (l-direction) and r-axis (m-direction).
///
/// Decomposition dimensions are driven by SimConfig::mpi_dims_l / mpi_dims_m.
/// Set mpi_dims_m = 1 in config.yaml for pure-MPI runs to keep the full M
/// range on each rank (better vectorisation, no column packing overhead).
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
    [[nodiscard]] auto IsLLoBoundary() const -> bool { return coords[0] == 0; }
    [[nodiscard]] auto IsLHiBoundary() const -> bool { return coords[0] == dims[0] - 1; }
    [[nodiscard]] auto IsMLoBoundary() const -> bool { return coords[1] == 0; }
    [[nodiscard]] auto IsMHiBoundary() const -> bool { return coords[1] == dims[1] - 1; }

    /// Initialises MPI and computes 2-D domain decomposition.
    /// Decomposition dimensions are taken from cfg.mpi_dims_l / mpi_dims_m
    /// (0 = let MPI_Dims_create decide).
    MPIManager(int& argc, char**& argv, const SimConfig& cfg);

    /// Frees the Cartesian communicator and calls MPI_Finalize.
    ~MPIManager();

    // Non-copyable, non-movable
    MPIManager(const MPIManager&)                    = delete;
    auto operator=(const MPIManager&) -> MPIManager& = delete;

    [[nodiscard]] auto Wtime() const -> double { return MPI_Wtime(); }

    /// Exchange one layer of ghost cells in all four Cartesian directions for
    /// a single 2-D array.
    ///
    /// Prefer exchange_ghosts_batch() when exchanging multiple arrays at once.
    ///
    /// Column buffers (m-direction, packed sends):
    ///   col_sl / col_sr : send buffers, size >= local_L_with_ghosts
    ///   col_rl / col_rr : recv buffers, size >= local_L_with_ghosts
    void ExchangeGhosts(double** arr,
                         double* col_sl, double* col_sr,
                         double* col_rl, double* col_rr) const;

    /// Exchange ghost cells for N arrays in a single non-blocking round.
    ///
    /// @param arrs      array of N double** pointers
    /// @param n         number of arrays (typically 18)
    /// @param col_bufs  scratch space — resized automatically each call.
    ///                  Reused across calls to avoid repeated allocation.
    void ExchangeGhostsBatch(double** const* arrs, int n,
                               std::vector<double>& col_bufs) const;
};
