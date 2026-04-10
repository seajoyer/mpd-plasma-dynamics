#pragma once

#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Layer-1 solver verification checks.
///
/// Three independent checks, intended to be run in this order:
///
///   1. Ghost-exchange round-trip  (CheckGhostExchange)
///      Fills a scratch array with unique cell identifiers, exchanges
///      ghosts, and verifies every received value against the expected
///      identifier of the sending cell.  A FAIL here is a hard MPI bug
///      (bad packing, wrong neighbour tag, wrong index arithmetic) and
///      must be fixed before trusting any physics result.
///      → Run once at startup, before the time loop.
///
///   2. Conserved-integral tracking  (ComputeIntegrals / PrintIntegrals /
///                                    ReportDrift)
///      Computes globally-integrated mass, momenta (z, r, φ), energy,
///      and magnetic flux by summing the conservative u_* arrays over
///      all interior cells.  For open BCs the integrals drift steadily;
///      use this not to enforce conservation but to detect sudden
///      anomalous jumps that signal NaN onset or a stencil bug.
///      → Compute reference at step 0; append rows every N steps; call
///        ReportDrift at the end of the run.
///
///   3. Radial-symmetry check  (CheckRadialSymmetry)
///      Computes the maximum normalised radial density gradient across
///      all interior cells.  Most useful after a radially-uniform
///      initial condition: the result should remain ~0 for a 1-D test
///      and grow noticeably if a symmetry-breaking bug is present.
///      → Run once right after field initialisation.
namespace Verification {

// ─── 1. Conserved integrals ──────────────────────────────────────────────

struct ConservedIntegrals {
    double mass{};
    double momentum_z{};
    double momentum_r{};
    double momentum_phi{};
    double energy{};
    double mag_flux_z{};
    double mag_flux_phi{};
};

/// Compute globally-integrated conserved quantities (MPI-collective).
///
/// Integration uses the conservative u_* arrays (which already carry the
/// cylindrical r factor) scaled by the 2-D cell area dr[l] * dz:
///
///   mass         ≈ Σ u_1[l][m] · dr[l] · dz      (= Σ ρ·r · dA)
///   momentum_z   ≈ Σ u_2[l][m] · dr[l] · dz
///   momentum_r   ≈ Σ u_3[l][m] · dr[l] · dz
///   momentum_phi ≈ Σ u_4[l][m] · dr[l] · dz
///   energy       ≈ Σ u_5[l][m] · dr[l] · dz
///   mag_flux_z   ≈ Σ u_7[l][m] · dr[l] · dz      (= Σ H_z·r · dA)
///   mag_flux_phi ≈ Σ u_6[l][m] · dr[l] · dz      (= Σ H_φ   · dA)
///
/// The 2π azimuthal factor is dropped consistently — it cancels when
/// comparing values across time steps.
auto ComputeIntegrals(const Fields& f, const Grid& g,
                                    const SimConfig& cfg,
                                    const MPIManager& mpi) -> ConservedIntegrals;

/// Print a column-header line for the integral table.
/// Call once before the first PrintIntegrals() row.  No-op on non-zero ranks.
void PrintIntegralsHeader(int rank);

/// Append one row to the running integral table.  No-op on non-zero ranks.
void PrintIntegrals(const ConservedIntegrals& q, int step, double t, int rank);

/// Print a per-quantity breakdown of the relative drift from a reference
/// snapshot and return the largest relative drift across all seven quantities.
///
/// Relative drift for quantity q:
///   drift(q) = |cur(q) − ref(q)| / (|ref(q)| + ε)
///
/// For a closed domain this should stay near machine epsilon.
/// For open BCs (inflow/outflow) a steady drift is expected; a sudden
/// increase signals a physics or numerics problem.
///
/// No-op printing on non-zero ranks; drift value returned on all ranks.
auto ReportDrift(const ConservedIntegrals& ref,
                   const ConservedIntegrals& cur,
                   int step, double t, int rank) -> double;

// ─── 2. Radial-symmetry check ────────────────────────────────────────────

/// Compute the maximum normalised radial density difference between adjacent
/// interior cells across all MPI ranks (MPI-collective):
///
///   max over (l, m) of  |ρ[l][m+1] − ρ[l][m]| / (0.5 * (ρ[l][m+1] + ρ[l][m]))
///
/// Interpretation:
///   • Near 0  — the density field is radially uniform (expected for a 1-D IC).
///   • Non-zero — radial structure is present (expected for the full thruster
///                configuration; note this as a baseline, not an error).
///
/// Prints one diagnostic line on rank 0.
auto CheckRadialSymmetry(const Fields& f, const MPIManager& mpi) -> double;

// ─── 3. Ghost-exchange round-trip ────────────────────────────────────────

/// Verify the batched ghost exchange (MPIManager::ExchangeGhostsBatch).
///
/// Algorithm:
///   1. Allocate a scratch Array2D with the local ghost-inclusive dimensions.
///   2. Fill every interior cell with a unique floating-point identity:
///        value(l_local, m_local) = (l_global + 1) * stride + (m_global + 1)
///      where stride = M_max + 10 prevents collisions between l-rows.
///      Ghost cells remain at zero so any un-exchanged ghost is immediately
///      distinguishable.
///   3. Call ExchangeGhostsBatch for this single array.
///   4. For each active neighbour direction, verify every received ghost
///      value against the expected identity of the sending interior cell.
///      Corner cells (where two ghost regions overlap) are excluded because
///      the sending rank stores zero there.
///
/// Prints PASS/FAIL + error count on rank 0.
/// Returns true on all ranks if the exchange is correct everywhere.
auto CheckGhostExchange(MPIManager& mpi, const SimConfig& cfg) -> bool;

} // namespace Verification
