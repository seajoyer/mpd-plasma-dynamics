#pragma once

#include <vector>
#include "config.hpp"
#include "face_bc.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Owns the numerical time-stepping algorithm:
///   1. Ghost-cell exchange in all four Cartesian directions (MPI)
///   2. Lax–Friedrichs central update for interior cells
///   3. Boundary conditions — dispatched through four FaceBC objects
///   4. Physical-variable reconstruction from conservative u
///   5. Advance u0 ← u
class Solver {
public:
    Solver(const SimConfig& cfg, const MPIManager& mpi,
           const Grid& grid, Fields& f);

    /// Execute one complete time step using the supplied dt.
    void Advance(double dt);

private:
    const SimConfig&  cfg_;
    const MPIManager& mpi_;
    const Grid&       grid_;
    Fields&           f_;

    double current_dt_{0.0};

    // ---- boundary conditions (one per Cartesian face) --------------------
    FaceBC bc_l_lo_;   ///< z = 0  face
    FaceBC bc_l_hi_;   ///< z = L  face
    FaceBC bc_m_lo_;   ///< r = inner face (may have multiple segments)
    FaceBC bc_m_hi_;   ///< r = outer face

    // ---- MPI scratch buffer + in-flight exchange handle -------------------
    std::vector<double> col_batch_buf_;
    MPIManager::GhostExchangeHandle ghost_handle_;

    // ---- cached constants (set once in the constructor) -------------------
    // The LF central update writes over the boundary strip on M-boundary
    // ranks where the BC will overwrite it anyway, so we skip those strips.
    // These offsets never change after MPI decomposition is fixed at start-up.
    int    m_lo_bc_{1};              ///< lowest interior m updated by the LF kernel
    int    m_hi_bc_{1};              ///< highest interior m updated by the LF kernel
    const double** r_ptr_{nullptr};  ///< grid_.r.Raw()  — bound once
    const double*  dr_ptr_{nullptr};  ///< grid_.dr.data()  — bound once

    // ---- sub-steps ----
    void PostGhostExchange();
    void FinishGhostExchange();
    void ComputeCentralUpdateRange(int l_lo, int l_hi, int m_lo, int m_hi);
    void UpdateCentralPhysical();
};
