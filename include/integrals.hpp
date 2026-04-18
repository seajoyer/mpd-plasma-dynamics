#pragma once

#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Outlet-plane physical diagnostics: mass flux and thrust.
///
/// Both quantities are integrals over the axisymmetric outlet cross-section
/// (the last axial column, at z = L − dz, i.e. global l-index L_max − 1):
///
///   MassFlux = ∫_{r_inner}^{r_outer}  ρ · v_z · 2π r · dr
///   Thrust   = ∫_{r_inner}^{r_outer}  (ρ v_z² + p + |H|²/(8π)) · 2π r · dr
///
/// where |H|² = H_z² + H_r² + H_phi².
///
/// Unit convention
/// ───────────────
/// The 1/(8π) factor in the magnetic-pressure term is the Gaussian-cgs
/// form.  It matches the thrust stress tensor of Villani (1982) /
/// Tkachenko et al. (2023).  Note that the solver's running quantity
/// P = p + ½·|H|² uses a different (rationalised) normalisation internally;
/// the Thrust expression below uses the 1/(8π) form directly because that
/// is the physically meaningful nozzle-exit stress.  If you change unit
/// conventions elsewhere, revisit GetThrust().
///
/// Numerical scheme
/// ────────────────
/// Composite trapezoid rule expressed as a *weighted sum over global
/// nodes*:
///
///   I ≈ Σ_{m=0}^{M_max}  w_m · F_m · dr      F_m = (integrand) · 2π · r_m
///
///   w_0 = w_{M_max} = ½     (endpoints, inner and outer walls)
///   w_m             = 1     for 1 ≤ m ≤ M_max − 1   (interior nodes)
///
/// This replaces the ad-hoc formula in the project's predecessor, which
/// double-counted interior nodes and omitted the trapezoid ½ factor.
///
/// MPI handling
/// ────────────
/// The outlet plane is owned only by ranks at coords[0] == dims[0] − 1
/// (IsLHiBoundary()).  Because the m-direction decomposition is
/// non-overlapping — each global m-node belongs to exactly one rank — the
/// node-weighted formulation above is decomposition-agnostic:
///
///   • Works correctly for any mpi_dims_m (1-D or 2-D decomposition).
///   • Produces the exact same floating-point result as a serial trapezoid
///     rule, modulo summation-order rounding in MPI_Allreduce.
///   • No "stitching" correction is needed at m-direction rank boundaries.
///
/// The functions are MPI-collective: every rank in MPI_COMM_WORLD must
/// call them.  Non-outlet ranks contribute zero to the internal reduction;
/// the final value is broadcast so all ranks return the same number.
namespace Diagnostics {

/// Integrate ρ · v_z · 2π r dr over the outlet cross-section.
///
/// Collective: must be called by every rank in MPI_COMM_WORLD.
/// Returns the same value on every rank.
auto GetMassFlux(const Fields& f, const Grid& g,
                 const SimConfig& cfg, const MPIManager& mpi) -> double;

/// Integrate (ρ v_z² + p + |H|²/(8π)) · 2π r dr over the outlet cross-section.
///
/// Collective: must be called by every rank in MPI_COMM_WORLD.
/// Returns the same value on every rank.
auto GetThrust(const Fields& f, const Grid& g,
               const SimConfig& cfg, const MPIManager& mpi) -> double;

} // namespace Diagnostics
