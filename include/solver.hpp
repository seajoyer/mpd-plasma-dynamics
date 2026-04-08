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
///
/// Boundary conditions are fully data-driven: each FaceBC holds an ordered
/// list of BCSegment objects, one per contiguous region of the face.  The
/// configuration is read from cfg (which was populated from config.yaml).
///
/// To add a new BC type, implement IBoundaryCondition, register it in
/// register_all_bcs() (src/bc_registry.cpp), and reference the name in
/// config.yaml — no changes to Solver are needed.
class Solver {
public:
    Solver(const SimConfig& cfg, const MPIManager& mpi,
           const Grid& grid, Fields& f);

    /// Execute one complete time step using the supplied dt.
    void advance(double dt);

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

    // ---- MPI scratch buffer -----------------------------------------------
    std::vector<double> col_batch_buf_;

    // ---- sub-steps ----
    void exchange_all_ghosts();
    void compute_central_update();
    void update_central_physical();
};
