#include "integrals.hpp"

#include <mpi.h>

#include <cmath>

namespace Diagnostics {

namespace {

// ──────────────────────────────────────────────────────────────────────
//  Outlet-plane integral using the composite trapezoid rule, correct
//  for arbitrary 2-D MPI decomposition (both l-axis and m-axis split).
//
//  Math
//  ────
//  The global composite trapezoid rule over M_max cells (M_max + 1 nodes
//  m_global = 0 … M_max) can be written as a *weighted sum over nodes*:
//
//      I ≈ Σ_{m_g = 0}^{M_max}  w(m_g) · F(m_g) · dr
//
//  with node weights
//
//      w(0) = w(M_max) = ½      (endpoints)
//      w(m)             = 1     for 1 ≤ m ≤ M_max − 1
//
//  and integrand F(m_g) = (physical integrand) · 2π · r  at node m_g.
//
//  Why this is parallel-friendly
//  ─────────────────────────────
//  Every global m-node is owned by **exactly one** rank in the current
//  decomposition (m_start … m_end are non-overlapping, no duplicates),
//  so each outlet rank can independently compute its partial sum over
//  its own interior m-nodes using the correct global-index weight, and
//  a single MPI_Allreduce produces the exact global integral — no
//  "stitching" error at rank boundaries, no special handling for
//  mpi_dims_m = 1 vs mpi_dims_m > 1.
//
//  Numerical equivalence to the serial trapezoid rule is exact in
//  floating point modulo the (order-dependent) summation rounding;
//  there is no additional discretisation error introduced by the
//  decomposition.
// ──────────────────────────────────────────────────────────────────────
template <typename Integrand>
auto IntegrateOutlet(const Fields& /*f*/, const Grid& g,
                     const SimConfig& cfg, const MPIManager& mpi,
                     Integrand integrand) -> double {
    double local_sum = 0.0;

    // Only ranks at the axial high-boundary own the outlet plane.
    //
    // Global l-index L_max − 1 corresponds to local l = mpi.local_L on
    // those ranks (because l_end == L_max − 1 for the last rank in dim 0,
    // and m_local = m_global − m_start + 1 ⇒ local_L = l_end − l_start + 1).
    if (mpi.IsLHiBoundary()) {
        const int    l      = mpi.local_L;
        const double dr_out = g.dr[l];

        // Walk this rank's interior m-nodes and accumulate their
        // weighted contribution to the *global* trapezoid sum.
        //
        // local_M interior cells ↔ global indices m_start … m_end.
        for (int m_local = 1; m_local <= mpi.local_M; ++m_local) {
            const int m_global = mpi.m_start + m_local - 1;

            // Node weight in the global composite trapezoid rule.
            // Endpoints m_global = 0 (inner wall) and m_global = M_max
            // (outer wall) get ½; all interior nodes get 1.
            const double w = (m_global == 0 || m_global == cfg.M_max)
                                 ? 0.5
                                 : 1.0;

            const double F = integrand(l, m_local) * 2.0 * M_PI * g.r[l][m_local];
            local_sum += w * F * dr_out;
        }
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_sum;
}

} // namespace

// ──────────────────────────────────────────────────────────────────────
//  Mass flux:  ∫ ρ · v_z · 2π r dr
// ──────────────────────────────────────────────────────────────────────
auto GetMassFlux(const Fields& f, const Grid& g,
                 const SimConfig& cfg, const MPIManager& mpi) -> double {
    return IntegrateOutlet(f, g, cfg, mpi,
        [&](int l, int m) -> double {
            return f.rho[l][m] * f.v_z[l][m];
        });
}

// ──────────────────────────────────────────────────────────────────────
//  Thrust:  ∫ (ρ v_z² + p + |H|²/(8π)) · 2π r dr
// ──────────────────────────────────────────────────────────────────────
auto GetThrust(const Fields& f, const Grid& g,
               const SimConfig& cfg, const MPIManager& mpi) -> double {
    constexpr double inv_8pi = 1.0 / (8.0 * M_PI);

    return IntegrateOutlet(f, g, cfg, mpi,
        [&](int l, int m) -> double {
            const double v_z2 = f.v_z[l][m] * f.v_z[l][m];
            const double H2   = f.H_z  [l][m] * f.H_z  [l][m]
                              + f.H_r  [l][m] * f.H_r  [l][m]
                              + f.H_phi[l][m] * f.H_phi[l][m];
            return f.rho[l][m] * v_z2 + f.p[l][m] + H2 * inv_8pi;
        });
}

} // namespace Diagnostics
