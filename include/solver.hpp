#pragma once

#include <vector>
#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Owns the numerical time-stepping algorithm:
///   1. Ghost-cell exchange in all four Cartesian directions (MPI)
///   2. Lax–Friedrichs central update for interior cells
///   3. Boundary conditions on the appropriate ranks
///   4. Physical-variable reconstruction from conservative u
///   5. Advance u0 ← u
///
/// The time step `dt` is passed explicitly to advance() so that the caller
/// (main loop) can drive adaptive time stepping independently of this class.
///
/// BC rank predicate:
///   apply_bc_left / apply_bc_right  → mpi.is_l_lo / l_hi boundary
///   apply_bc_upper / apply_bc_lower → mpi.is_m_hi / m_lo boundary
///
/// Inner-wall index convention (design decision 3):
///   On the m-lo boundary rank m_local=1 IS the inner wall cell.
///   Every reference to the first interior cell above the wall uses [m+1]
///   (= [2] when m=1), never a hardcoded [1].
class Solver {
public:
    Solver(const SimConfig& cfg, const MPIManager& mpi,
           const Grid& grid, Fields& f);

    /// Execute one complete time step using the supplied dt.
    ///
    /// When adaptive time stepping is active, the main loop computes dt via
    /// Diagnostics::compute_dt() after each step and passes the new value
    /// here.  When adaptive stepping is disabled the main loop simply passes
    /// cfg.dt unchanged every call.
    void advance(double dt);

private:
    const SimConfig&  cfg_;
    const MPIManager& mpi_;
    const Grid&       grid_;
    Fields&           f_;

    /// The dt value for the step currently being computed.
    /// Set at the top of advance() and read by all private sub-steps.
    double current_dt_{0.0};

    // ---- MPI scratch buffer for the batched ghost exchange -----------------
    // Holds packed column data for all 18 arrays × 4 directions.
    // Layout: [ send_lo | send_hi | recv_lo | recv_hi ] × nfields,
    // each segment being local_L_with_ghosts doubles.
    // Resized once on the first call; reused every subsequent step.
    std::vector<double> col_batch_buf_;

    // ---- sub-steps ----

    /// Exchange ghost cells for all 18 arrays (8 conservative + 10 physical)
    /// in a single non-blocking MPI round (one Waitall for everything).
    void exchange_all_ghosts();

    /// Lax–Friedrichs update for interior cells.
    void compute_central_update();

    /// Reconstruct physical vars from u for the central interior.
    void update_central_physical();

    // ---- boundary conditions ----
    void apply_bc_left();             ///< inflow BC at z = 0  (l-lo rank only)
    void apply_bc_right();            ///< outflow BC at z = L (l-hi rank only)
    void apply_bc_upper();            ///< outer-wall BC at m_local=local_M (m-hi rank only)
    void apply_bc_lower_inner();      ///< inner-wall BC, l ≤ L_end (m-lo rank only)
    void apply_bc_lower_outer();      ///< inner-wall free BC, l > L_end (m-lo rank only)

    /// Rebuild u from the physical values set by the BC routines at (l, m).
    inline void rebuild_u_from_physical(int l, int m);
};
