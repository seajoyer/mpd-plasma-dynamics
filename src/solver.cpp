#include "solver.hpp"

#include <cmath>
#include <algorithm>

// ============================================================
// Constructor
// ============================================================

Solver::Solver(const SimConfig& cfg, const MPIManager& mpi,
               const Grid& grid, Fields& f)
    : cfg_(cfg), mpi_(mpi), grid_(grid), f_(f),
      // Row buffers for l-direction ghost exchange (each row is contiguous).
      row_sl_(mpi.local_M_with_ghosts), row_sr_(mpi.local_M_with_ghosts),
      row_rl_(mpi.local_M_with_ghosts), row_rr_(mpi.local_M_with_ghosts),
      // Column buffers for m-direction ghost exchange (packed).
      col_sl_(mpi.local_L_with_ghosts), col_sr_(mpi.local_L_with_ghosts),
      col_rl_(mpi.local_L_with_ghosts), col_rr_(mpi.local_L_with_ghosts)
{}

// ============================================================
// Public entry point
// ============================================================

void Solver::advance() {
    exchange_all_ghosts();
    compute_central_update();
    update_central_physical();
    apply_bc_left();
    apply_bc_upper();
    apply_bc_lower_inner();
    apply_bc_lower_outer();
    apply_bc_right();
    // Full reconstruction: covers all cells set by either central update or BCs.
    f_.update_physical_from_u(grid_, cfg_, 1, mpi_.local_L, 1, mpi_.local_M);
    f_.copy_u_to_u0();
}

// ============================================================
// Ghost-cell exchange  (all 4 directions, all 18 arrays)
// ============================================================

void Solver::exchange_all_ghosts() {
    auto exchange = [&](double** arr) {
        mpi_.exchange_ghosts(arr,
            row_sl_.data(), row_sr_.data(), row_rl_.data(), row_rr_.data(),
            col_sl_.data(), col_sr_.data(), col_rl_.data(), col_rr_.data());
    };

    // 8 conservative arrays
    double** u0_ptrs[8] = {
        f_.u0_1.raw(), f_.u0_2.raw(), f_.u0_3.raw(), f_.u0_4.raw(),
        f_.u0_5.raw(), f_.u0_6.raw(), f_.u0_7.raw(), f_.u0_8.raw()
    };
    for (auto* arr : u0_ptrs) exchange(arr);

    // 10 physical arrays
    double** phys_ptrs[10] = {
        f_.rho.raw(),   f_.v_z.raw(),   f_.v_r.raw(),
        f_.v_phi.raw(), f_.e.raw(),     f_.p.raw(),
        f_.P.raw(),     f_.H_z.raw(),   f_.H_r.raw(),
        f_.H_phi.raw()
    };
    for (auto* arr : phys_ptrs) exchange(arr);
}

// ============================================================
// Lax–Friedrichs central update
// ============================================================

void Solver::compute_central_update() {
    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;
    const double dt   = cfg_.dt;
    const double dz   = cfg_.dz;

    // Skip wall cells owned by this rank:
    //   bottom rank (is_m_lo_boundary): m=1 is inner wall → start at m=2
    //   top    rank (is_m_hi_boundary): m=local_M is outer wall → end at local_M-1
    const int m_lo = mpi_.is_m_lo_boundary() ? 2 : 1;
    const int m_hi = mpi_.is_m_hi_boundary() ? local_M - 1 : local_M;

    // Raw double** aliases for the performance-critical inner loop.
    auto** u0_1 = f_.u0_1.raw();  auto** u0_2 = f_.u0_2.raw();
    auto** u0_3 = f_.u0_3.raw();  auto** u0_4 = f_.u0_4.raw();
    auto** u0_5 = f_.u0_5.raw();  auto** u0_6 = f_.u0_6.raw();
    auto** u0_7 = f_.u0_7.raw();  auto** u0_8 = f_.u0_8.raw();

    auto** u_1  = f_.u_1.raw();   auto** u_2  = f_.u_2.raw();
    auto** u_3  = f_.u_3.raw();   auto** u_4  = f_.u_4.raw();
    auto** u_5  = f_.u_5.raw();   auto** u_6  = f_.u_6.raw();
    auto** u_7  = f_.u_7.raw();   auto** u_8  = f_.u_8.raw();

    auto** rho   = f_.rho.raw();   auto** v_z  = f_.v_z.raw();
    auto** v_r   = f_.v_r.raw();   auto** v_phi= f_.v_phi.raw();
    auto** p     = f_.p.raw();     auto** P    = f_.P.raw();
    auto** H_z   = f_.H_z.raw();   auto** H_r  = f_.H_r.raw();
    auto** H_phi = f_.H_phi.raw();
    auto** r     = grid_.r.raw();
    const double* dr = grid_.dr.data();

    #pragma omp parallel for collapse(2)
    for (int l = 1; l <= local_L; ++l) {
        for (int m = m_lo; m <= m_hi; ++m) {
            const double dr_l = dr[l];

            // u1: mass conservation (rho * r)
            u_1[l][m] =
                0.25 * (u0_1[l+1][m] + u0_1[l-1][m] + u0_1[l][m+1] + u0_1[l][m-1])
                + dt * (-(u0_1[l+1][m]*v_z[l+1][m] - u0_1[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_1[l][m+1]*v_r[l][m+1]  - u0_1[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            // u2: z-momentum
            u_2[l][m] =
                0.25 * (u0_2[l+1][m] + u0_2[l-1][m] + u0_2[l][m+1] + u0_2[l][m-1])
                + dt * (( (H_z[l+1][m]*H_z[l+1][m] - P[l+1][m])*r[l+1][m]
                         -(H_z[l-1][m]*H_z[l-1][m] - P[l-1][m])*r[l-1][m]) / (2*dz)
                        +( H_z[l][m+1]*H_r[l][m+1]*r[l][m+1]
                          -H_z[l][m-1]*H_r[l][m-1]*r[l][m-1]) / (2*dr_l)
                        -(u0_2[l+1][m]*v_z[l+1][m] - u0_2[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_2[l][m+1]*v_r[l][m+1]  - u0_2[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            // u3: r-momentum
            u_3[l][m] =
                0.25 * (u0_3[l+1][m] + u0_3[l-1][m] + u0_3[l][m+1] + u0_3[l][m-1])
                + dt * ((rho[l][m]*v_phi[l][m]*v_phi[l][m] + P[l][m] - H_phi[l][m]*H_phi[l][m])
                        +( H_z[l+1][m]*H_r[l+1][m]*r[l+1][m]
                          -H_z[l-1][m]*H_r[l-1][m]*r[l-1][m]) / (2*dz)
                        +( (H_r[l][m+1]*H_r[l][m+1] - P[l][m+1])*r[l][m+1]
                          -(H_r[l][m-1]*H_r[l][m-1] - P[l][m-1])*r[l][m-1]) / (2*dr_l)
                        -(u0_3[l+1][m]*v_z[l+1][m] - u0_3[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_3[l][m+1]*v_r[l][m+1]  - u0_3[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            // u4: phi-momentum
            u_4[l][m] =
                0.25 * (u0_4[l+1][m] + u0_4[l-1][m] + u0_4[l][m+1] + u0_4[l][m-1])
                + dt * ((-rho[l][m]*v_r[l][m]*v_phi[l][m] + H_phi[l][m]*H_r[l][m])
                        +( H_phi[l+1][m]*H_z[l+1][m]*r[l+1][m]
                          -H_phi[l-1][m]*H_z[l-1][m]*r[l-1][m]) / (2*dz)
                        +( H_phi[l][m+1]*H_r[l][m+1]*r[l][m+1]
                          -H_phi[l][m-1]*H_r[l][m-1]*r[l][m-1]) / (2*dr_l)
                        -(u0_4[l+1][m]*v_z[l+1][m] - u0_4[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_4[l][m+1]*v_r[l][m+1]  - u0_4[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            // u5: internal energy
            u_5[l][m] =
                0.25 * (u0_5[l+1][m] + u0_5[l-1][m] + u0_5[l][m+1] + u0_5[l][m-1])
                + dt * (-p[l][m] * ((v_z[l+1][m]*r[l+1][m] - v_z[l-1][m]*r[l-1][m]) / (2*dz)
                                   +(v_r[l][m+1]*r[l][m+1]  - v_r[l][m-1]*r[l][m-1])  / (2*dr_l))
                        -(u0_5[l+1][m]*v_z[l+1][m] - u0_5[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_5[l][m+1]*v_r[l][m+1]  - u0_5[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            // u6: H_phi (toroidal field)
            u_6[l][m] =
                0.25 * (u0_6[l+1][m] + u0_6[l-1][m] + u0_6[l][m+1] + u0_6[l][m-1])
                + dt * ((H_z[l+1][m]*v_phi[l+1][m] - H_z[l-1][m]*v_phi[l-1][m]) / (2*dz)
                       +(H_r[l][m+1]*v_phi[l][m+1]  - H_r[l][m-1]*v_phi[l][m-1])  / (2*dr_l)
                        -(u0_6[l+1][m]*v_z[l+1][m] - u0_6[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_6[l][m+1]*v_r[l][m+1]  - u0_6[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            // u7: H_z * r (axial flux)
            u_7[l][m] =
                0.25 * (u0_7[l+1][m] + u0_7[l-1][m] + u0_7[l][m+1] + u0_7[l][m-1])
                + dt * ((H_r[l][m+1]*v_z[l][m+1]*r[l][m+1] - H_r[l][m-1]*v_z[l][m-1]*r[l][m-1]) / (2*dr_l)
                        -(u0_7[l][m+1]*v_r[l][m+1]  - u0_7[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            // u8: H_r * r (radial flux)
            u_8[l][m] =
                0.25 * (u0_8[l+1][m] + u0_8[l-1][m] + u0_8[l][m+1] + u0_8[l][m-1])
                + dt * ((H_z[l+1][m]*v_r[l+1][m]*r[l+1][m] - H_z[l-1][m]*v_r[l-1][m]*r[l-1][m]) / (2*dz)
                        -(u0_8[l+1][m]*v_z[l+1][m] - u0_8[l-1][m]*v_z[l-1][m]) / (2*dz));
        }
    }
}

void Solver::update_central_physical() {
    const int m_lo = mpi_.is_m_lo_boundary() ? 2 : 1;
    const int m_hi = mpi_.is_m_hi_boundary() ? mpi_.local_M - 1 : mpi_.local_M;
    f_.update_physical_from_u(grid_, cfg_, 1, mpi_.local_L, m_lo, m_hi);
}

// ============================================================
// Inline helper: rebuild conservative u from physical at (l, m)
// ============================================================

inline void Solver::rebuild_u_from_physical(int l, int m) {
    f_.u_1[l][m] = f_.rho  [l][m] * grid_.r[l][m];
    f_.u_2[l][m] = f_.rho  [l][m] * f_.v_z  [l][m] * grid_.r[l][m];
    f_.u_3[l][m] = f_.rho  [l][m] * f_.v_r  [l][m] * grid_.r[l][m];
    f_.u_4[l][m] = f_.rho  [l][m] * f_.v_phi[l][m] * grid_.r[l][m];
    f_.u_5[l][m] = f_.rho  [l][m] * f_.e    [l][m] * grid_.r[l][m];
    f_.u_6[l][m] = f_.H_phi[l][m];
    f_.u_7[l][m] = f_.H_z  [l][m] * grid_.r[l][m];
    f_.u_8[l][m] = f_.H_r  [l][m] * grid_.r[l][m];
}

// ============================================================
// Boundary conditions
// ============================================================

// ---- Inflow BC (z = 0, l-lo boundary rank) -------------------------

void Solver::apply_bc_left() {
    if (!mpi_.is_l_lo_boundary()) return;

    const double gamma = cfg_.gamma;
    const double beta  = cfg_.beta;
    const double H_z0  = cfg_.H_z0;
    const double r_0   = grid_.r_0;
    const int    local_M = mpi_.local_M;

    #pragma omp parallel for
    for (int m = 1; m <= local_M; ++m) {
        constexpr int l = 1;
        f_.rho  [l][m] = 1.0;
        f_.v_phi[l][m] = 0.0;
        // v_z derived from z-momentum of the updated neighbour at l=2.
        f_.v_z  [l][m] = f_.u_2[2][m] / (f_.rho[l][m] * grid_.r[l][m]);
        f_.v_r  [l][m] = 0.0;
        f_.H_phi[l][m] = r_0 / grid_.r[l][m];
        f_.H_z  [l][m] = H_z0;
        f_.H_r  [l][m] = 0.0;
        f_.e    [l][m] = beta / (2.0 * (gamma - 1.0))
                         * std::pow(f_.rho[l][m], gamma - 1.0);

        rebuild_u_from_physical(l, m);
    }
}

// ---- Outflow BC (z = L, l-hi boundary rank) ------------------------

void Solver::apply_bc_right() {
    if (!mpi_.is_l_hi_boundary()) return;

    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;

    #pragma omp parallel for
    for (int m = 1; m <= local_M; ++m) {
        f_.u_1[local_L][m] = f_.u_1[local_L - 1][m];
        f_.u_2[local_L][m] = f_.u_2[local_L - 1][m];
        f_.u_3[local_L][m] = f_.u_3[local_L - 1][m];
        f_.u_4[local_L][m] = f_.u_4[local_L - 1][m];
        f_.u_5[local_L][m] = f_.u_5[local_L - 1][m];
        f_.u_6[local_L][m] = f_.u_6[local_L - 1][m];
        f_.u_7[local_L][m] = f_.u_7[local_L - 1][m];
        f_.u_8[local_L][m] = f_.u_8[local_L - 1][m];
    }
}

// ---- Outer-wall BC (m = M_max, m-hi boundary rank) -----------------

void Solver::apply_bc_upper() {
    if (!mpi_.is_m_hi_boundary()) return;

    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;   // m_local=local_M  ≡  m_global=M_max

    #pragma omp parallel for
    for (int l = 1; l <= local_L; ++l) {
        const int m = local_M;
        f_.rho  [l][m] = f_.rho  [l][m-1];
        f_.v_z  [l][m] = f_.v_z  [l][m-1];
        f_.v_r  [l][m] = f_.v_z  [l][m] * grid_.r_z[l][m];
        f_.v_phi[l][m] = f_.v_phi[l][m-1];
        f_.e    [l][m] = f_.e    [l][m-1];
        f_.H_phi[l][m] = f_.H_phi[l][m-1];
        f_.H_z  [l][m] = f_.H_z  [l][m-1];
        f_.H_r  [l][m] = f_.H_z  [l][m] * grid_.r_z[l][m];

        rebuild_u_from_physical(l, m);
    }
}

// ---- Inner-wall BC, convergent section (l_global ≤ L_end, m-lo rank) ---
//
// Design decision 3: the inner wall lives at m_local=1 on the bottom rank.
// All references to "the first interior cell above the wall" use [m+1] = [2].

void Solver::apply_bc_lower_inner() {
    if (!mpi_.is_m_lo_boundary()) return;

    const int L_end   = cfg_.L_end;
    const int l_start = mpi_.l_start;
    const int l_end_g = mpi_.l_end;

    if (l_start > L_end || l_end_g < 1) return;

    const int L_end_in_domain = std::min(L_end, l_end_g);
    const int local_L_end_rel = L_end_in_domain - l_start + 1;

    for (int l = 1; l <= local_L_end_rel; ++l) {
        const int l_global = l_start + l - 1;
        if (l_global < 1 || l_global > L_end) continue;

        // m=1  : inner wall cell  (m_global = 0)
        // m+1=2: first interior cell above the wall
        const int m = 1;
        f_.rho  [l][m] = f_.rho  [l][m+1];
        f_.v_z  [l][m] = f_.v_z  [l][m+1];
        f_.v_r  [l][m] = f_.v_z  [l][m+1] * grid_.r_z[l][m+1];
        f_.v_phi[l][m] = f_.v_phi[l][m+1];
        f_.e    [l][m] = f_.e    [l][m+1];
        f_.H_phi[l][m] = f_.H_phi[l][m+1];
        f_.H_z  [l][m] = f_.H_z  [l][m+1];
        f_.H_r  [l][m] = f_.H_z  [l][m+1] * grid_.r_z[l][m+1];

        rebuild_u_from_physical(l, m);
    }
}

// ---- Inner-wall BC, divergent section (l_global > L_end, m-lo rank) ----
//
// One-sided Lax–Friedrichs-like update at the inner wall.
// The stencil uses [m] (wall) and [m+1] (first interior cell) in the
// r-direction only; there is no [m-1] term.
//
// Design decision 3: [m+1] = [2] when m=1.  The single hardcoded [1] in the
// mass-flux term (v_r[l][1] in the original code) is replaced by [m+1].

void Solver::apply_bc_lower_outer() {
    if (!mpi_.is_m_lo_boundary()) return;

    const int local_L = mpi_.local_L;
    const int L_end   = cfg_.L_end;
    const int L_max   = cfg_.L_max_global;
    const int l_start = mpi_.l_start;

    const double dt  = cfg_.dt;
    const double dz  = cfg_.dz;

    auto** u0_1 = f_.u0_1.raw(); auto** u0_2 = f_.u0_2.raw();
    auto** u0_5 = f_.u0_5.raw(); auto** u0_7 = f_.u0_7.raw();
    auto** v_z  = f_.v_z.raw();
    auto** v_r  = f_.v_r.raw();  auto** p    = f_.p.raw();
    auto** P    = f_.P.raw();    auto** H_z  = f_.H_z.raw();
    auto** H_r  = f_.H_r.raw();
    auto** r    = grid_.r.raw();
    const double* dr = grid_.dr.data();

    // m=1 is the inner wall; m+1=2 is the first interior cell above it.
    constexpr int m = 1;

    #pragma omp parallel for
    for (int l = 1; l <= local_L; ++l) {
        const int l_global = l_start + l - 1;
        if (l_global <= L_end || l_global >= L_max) continue;

        const double dr_l = dr[l];

        // u1: mass conservation.
        // The advective flux in the r-direction uses v_r at [m+1] for the
        // "interior" side and v_r at [m+1] (= first interior) for the "wall"
        // side — matching the original one-sided axis stencil.
        f_.u_1[l][m] =
            (0.25 * (u0_1[l+1][m]/r[l+1][m] + u0_1[l-1][m]/r[l-1][m]
                   + u0_1[l][m+1]/r[l][m+1]  + u0_1[l][m]/r[l][m])
             + dt * (-(u0_1[l+1][m]/r[l+1][m]*v_z[l+1][m]
                      -u0_1[l-1][m]/r[l-1][m]*v_z[l-1][m]) / (2*dz)
                     -(u0_1[l][m+1]/r[l][m+1]*v_r[l][m+1]
                      -u0_1[l][m]/r[l][m]*v_r[l][m+1])       / dr_l))
            * r[l][m];

        // u2: z-momentum.
        f_.u_2[l][m] =
            (0.25 * (u0_2[l+1][m]/r[l+1][m] + u0_2[l-1][m]/r[l-1][m]
                   + u0_2[l][m+1]/r[l][m+1]  + u0_2[l][m]/r[l][m])
             + dt * (( (H_z[l+1][m]*H_z[l+1][m] - P[l+1][m])
                      -(H_z[l-1][m]*H_z[l-1][m] - P[l-1][m])) / (2*dz)
                    +( H_z[l][m+1]*H_r[l][m+1] - H_z[l][m]*H_r[l][m]) / dr_l
                     -(u0_2[l+1][m]/r[l+1][m]*v_z[l+1][m]
                      -u0_2[l-1][m]/r[l-1][m]*v_z[l-1][m]) / (2*dz)
                     -(u0_2[l][m+1]/r[l][m+1]*v_r[l][m+1]
                      -u0_2[l][m]/r[l][m]*v_r[l][m])           / dr_l))
            * r[l][m];

        // u3, u4: r- and phi-momentum vanish at the inner wall.
        f_.u_3[l][m] = 0.0;
        f_.u_4[l][m] = 0.0;

        // u5: internal energy.
        f_.u_5[l][m] =
            (0.25 * (u0_5[l+1][m]/r[l+1][m] + u0_5[l-1][m]/r[l-1][m]
                   + u0_5[l][m+1]/r[l][m+1]  + u0_5[l][m]/r[l][m])
             + dt * (-p[l][m] * ((v_z[l+1][m] - v_z[l-1][m]) / (2*dz)
                                +(v_r[l][m+1]  - v_r[l][m])    / dr_l)
                     -(u0_5[l+1][m]/r[l+1][m]*v_z[l+1][m]
                      -u0_5[l-1][m]/r[l-1][m]*v_z[l-1][m]) / (2*dz)
                     -(u0_5[l][m+1]/r[l][m+1]*v_r[l][m+1]
                      -u0_5[l][m]/r[l][m]*v_r[l][m])           / dr_l))
            * r[l][m];

        // u6: H_phi vanishes at the inner wall (axial symmetry).
        f_.u_6[l][m] = 0.0;

        // u7: H_z * r (axial flux).
        f_.u_7[l][m] =
            (0.25 * (u0_7[l+1][m]/r[l+1][m] + u0_7[l-1][m]/r[l-1][m]
                   + u0_7[l][m+1]/r[l][m+1]  + u0_7[l][m]/r[l][m])
             + dt * ((H_r[l][m+1]*v_z[l][m+1] - H_r[l][m]*v_z[l][m]) / dr_l
                     -(u0_7[l][m+1]/r[l][m+1]*v_r[l][m+1]
                      -u0_7[l][m]/r[l][m]*v_r[l][m])                   / dr_l))
            * r[l][m];

        // u8: H_r vanishes at the inner wall.
        f_.u_8[l][m] = 0.0;
    }
}
