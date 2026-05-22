#pragma once

#include <vector>
#include "config.hpp"
#include "face_bc.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Owns the numerical time-stepping algorithm.
///
/// One call to Advance(dt) executes:
///
///   1. Ghost-cell exchange of u0_* (non-blocking, overlapped with step 2)
///   2. Lax–Friedrichs central update of the deep interior  (writes u_*)
///   3. Wait for ghost exchange
///   4. LF central update of the boundary strips that depend on ghost rings
///   5. Reconstruct *just the BC neighbour strips* from u_*  (~1 row/col each)
///   6. Apply four FaceBC objects (one per Cartesian face)
///   7. Reconstruct the BC-owned boundary cells from u_*  (one row/col each)
///   8. Pointer-swap u_* ↔ u0_*
///
class Solver {
public:
    Solver(const SimConfig& cfg, const MPIManager& mpi,
           const Grid& grid, Fields& f);

    /// Execute one complete time step using the supplied dt.
    void Advance(double dt);

    /// Reconstruct all physical fields (rho, v_*, H_*, e) on the owned
    /// interior 1..local_L × 1..local_M from the post-swap conservative
    /// state u0_*.  Call once immediately before each diagnostic or I/O
    /// barrier that reads physical fields.
    ///
    /// Internally guarded by a dirty flag: the first call after Advance()
    /// performs the full reconstruction; subsequent calls until the next
    /// Advance() are no-ops.  This lets callers sprinkle SyncPhysicalState()
    /// defensively before each block that reads physicals without paying
    /// twice when several blocks fire on the same step.
    void SyncPhysicalState();

private:
    const SimConfig&  cfg_;
    const MPIManager& mpi_;
    const Grid&       grid_;
    Fields&           f_;

    double current_dt_{0.0};

    /// True after Advance() returns and before the next SyncPhysicalState()
    /// call.  Tracks whether the interior physicals reflect the current
    /// conservative state.  Boundary-strip physicals are always in sync.
    bool physical_dirty_{false};

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
    const double** r_ptr_{nullptr};  ///< grid_.r.Raw()   — bound once
    const double*  dr_ptr_{nullptr}; ///< grid_.dr.data() — bound once

    // ---- sub-steps ----
    void PostGhostExchange();
    void FinishGhostExchange();
    void ComputeCentralUpdateRange(int l_lo, int l_hi, int m_lo, int m_hi);

    /// Reconstruct the one-cell-thick neighbour strip that each owned BC
    /// will read.  Called between the LF kernel and the BC apply, so the
    /// freshly-written u_* values become visible as primitives to the BCs.
    void ReconstructBCNeighborStrips();

    /// Reconstruct the boundary cells that the FaceBCs just wrote u_* for.
    /// Keeps physicals on the four boundary strips in sync with the BC
    /// output for the next step's neighbour reads (at corner cells) and
    /// for diagnostics that may run immediately after Advance().
    void ReconstructBCBoundaryStrips();
};
