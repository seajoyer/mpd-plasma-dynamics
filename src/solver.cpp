#include "solver.hpp"

#include <algorithm>
#include <cmath>

// ============================================================
// Constructor — build FaceBC objects from config
// ============================================================

Solver::Solver(const SimConfig& cfg, const MPIManager& mpi,
               const Grid& grid, Fields& f)
    : cfg_(cfg), mpi_(mpi), grid_(grid), f_(f),
      bc_l_lo_(FaceBC::from_config(FaceBC::Face::L_LO, cfg.bc_l_lo)),
      bc_l_hi_(FaceBC::from_config(FaceBC::Face::L_HI, cfg.bc_l_hi)),
      bc_m_lo_(FaceBC::from_config(FaceBC::Face::M_LO, cfg.bc_m_lo)),
      bc_m_hi_(FaceBC::from_config(FaceBC::Face::M_HI, cfg.bc_m_hi))
{}

// ============================================================
// Public entry point
// ============================================================

void Solver::advance(double dt) {
    current_dt_ = dt;

    exchange_all_ghosts();
    compute_central_update();
    update_central_physical();

    // Apply BCs in a deterministic order:
    //   L_LO before M_LO so that the inflow corner cell (l=1,m=1) is set by
    //   the inflow BC, then protected from overwrite by the corner policy in
    //   FaceBC::apply() for M_LO.
    bc_l_lo_.apply(f_, grid_, cfg_, mpi_, dt);
    bc_l_hi_.apply(f_, grid_, cfg_, mpi_, dt);
    bc_m_hi_.apply(f_, grid_, cfg_, mpi_, dt);
    bc_m_lo_.apply(f_, grid_, cfg_, mpi_, dt);

    // Full physical reconstruction: covers all cells (central + BC cells).
    f_.update_physical_from_u(grid_, cfg_, 1, mpi_.local_L, 1, mpi_.local_M);
    f_.copy_u_to_u0();
}

// ============================================================
// Ghost-cell exchange (all 4 directions, all 18 arrays, one phase)
// ============================================================

void Solver::exchange_all_ghosts() {
    double** arrs[18] = {
        f_.u0_1.raw(), f_.u0_2.raw(), f_.u0_3.raw(), f_.u0_4.raw(),
        f_.u0_5.raw(), f_.u0_6.raw(), f_.u0_7.raw(), f_.u0_8.raw(),
        f_.rho.raw(),   f_.v_z.raw(),   f_.v_r.raw(),
        f_.v_phi.raw(), f_.e.raw(),     f_.p.raw(),
        f_.P.raw(),     f_.H_z.raw(),   f_.H_r.raw(),
        f_.H_phi.raw()
    };
    mpi_.exchange_ghosts_batch(arrs, 18, col_batch_buf_);
}

// ============================================================
// Lax–Friedrichs central update (interior cells)
// ============================================================

void Solver::compute_central_update() {
    const int    local_L = mpi_.local_L;
    const int    local_M = mpi_.local_M;
    const double dt      = current_dt_;
    const double dz      = cfg_.dz;

    // On M boundary ranks the first/last interior row of the M direction
    // belongs to a BC; skip it in the central update so the BC result is
    // not overwritten.
    const int m_lo = mpi_.is_m_lo_boundary() ? 2 : 1;
    const int m_hi = mpi_.is_m_hi_boundary() ? local_M - 1 : local_M;

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

            u_1[l][m] =
                0.25 * (u0_1[l+1][m] + u0_1[l-1][m] + u0_1[l][m+1] + u0_1[l][m-1])
                + dt * (-(u0_1[l+1][m]*v_z[l+1][m] - u0_1[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_1[l][m+1]*v_r[l][m+1]  - u0_1[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            u_2[l][m] =
                0.25 * (u0_2[l+1][m] + u0_2[l-1][m] + u0_2[l][m+1] + u0_2[l][m-1])
                + dt * (( (H_z[l+1][m]*H_z[l+1][m] - P[l+1][m])*r[l+1][m]
                         -(H_z[l-1][m]*H_z[l-1][m] - P[l-1][m])*r[l-1][m]) / (2*dz)
                        +( H_z[l][m+1]*H_r[l][m+1]*r[l][m+1]
                          -H_z[l][m-1]*H_r[l][m-1]*r[l][m-1]) / (2*dr_l)
                        -(u0_2[l+1][m]*v_z[l+1][m] - u0_2[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_2[l][m+1]*v_r[l][m+1]  - u0_2[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            u_3[l][m] =
                0.25 * (u0_3[l+1][m] + u0_3[l-1][m] + u0_3[l][m+1] + u0_3[l][m-1])
                + dt * ((rho[l][m]*v_phi[l][m]*v_phi[l][m] + P[l][m] - H_phi[l][m]*H_phi[l][m])
                        +( H_z[l+1][m]*H_r[l+1][m]*r[l+1][m]
                          -H_z[l-1][m]*H_r[l-1][m]*r[l-1][m]) / (2*dz)
                        +( (H_r[l][m+1]*H_r[l][m+1] - P[l][m+1])*r[l][m+1]
                          -(H_r[l][m-1]*H_r[l][m-1] - P[l][m-1])*r[l][m-1]) / (2*dr_l)
                        -(u0_3[l+1][m]*v_z[l+1][m] - u0_3[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_3[l][m+1]*v_r[l][m+1]  - u0_3[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            u_4[l][m] =
                0.25 * (u0_4[l+1][m] + u0_4[l-1][m] + u0_4[l][m+1] + u0_4[l][m-1])
                + dt * ((-rho[l][m]*v_r[l][m]*v_phi[l][m] + H_phi[l][m]*H_r[l][m])
                        +( H_phi[l+1][m]*H_z[l+1][m]*r[l+1][m]
                          -H_phi[l-1][m]*H_z[l-1][m]*r[l-1][m]) / (2*dz)
                        +( H_phi[l][m+1]*H_r[l][m+1]*r[l][m+1]
                          -H_phi[l][m-1]*H_r[l][m-1]*r[l][m-1]) / (2*dr_l)
                        -(u0_4[l+1][m]*v_z[l+1][m] - u0_4[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_4[l][m+1]*v_r[l][m+1]  - u0_4[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            u_5[l][m] =
                0.25 * (u0_5[l+1][m] + u0_5[l-1][m] + u0_5[l][m+1] + u0_5[l][m-1])
                + dt * (-p[l][m] * ((v_z[l+1][m]*r[l+1][m] - v_z[l-1][m]*r[l-1][m]) / (2*dz)
                                   +(v_r[l][m+1]*r[l][m+1]  - v_r[l][m-1]*r[l][m-1])  / (2*dr_l))
                        -(u0_5[l+1][m]*v_z[l+1][m] - u0_5[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_5[l][m+1]*v_r[l][m+1]  - u0_5[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            u_6[l][m] =
                0.25 * (u0_6[l+1][m] + u0_6[l-1][m] + u0_6[l][m+1] + u0_6[l][m-1])
                + dt * ((H_z[l+1][m]*v_phi[l+1][m] - H_z[l-1][m]*v_phi[l-1][m]) / (2*dz)
                       +(H_r[l][m+1]*v_phi[l][m+1]  - H_r[l][m-1]*v_phi[l][m-1])  / (2*dr_l)
                        -(u0_6[l+1][m]*v_z[l+1][m] - u0_6[l-1][m]*v_z[l-1][m]) / (2*dz)
                        -(u0_6[l][m+1]*v_r[l][m+1]  - u0_6[l][m-1]*v_r[l][m-1])  / (2*dr_l));

            u_7[l][m] =
                0.25 * (u0_7[l+1][m] + u0_7[l-1][m] + u0_7[l][m+1] + u0_7[l][m-1])
                + dt * ((H_r[l][m+1]*v_z[l][m+1]*r[l][m+1] - H_r[l][m-1]*v_z[l][m-1]*r[l][m-1]) / (2*dr_l)
                        -(u0_7[l][m+1]*v_r[l][m+1]  - u0_7[l][m-1]*v_r[l][m-1])  / (2*dr_l));

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
