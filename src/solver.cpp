#include "solver.hpp"

#include <cmath>

// ============================================================
// Constructor — build FaceBC objects from config
// ============================================================

Solver::Solver(const SimConfig& cfg, const MPIManager& mpi,
               const Grid& grid, Fields& f)
    : cfg_(cfg), mpi_(mpi), grid_(grid), f_(f),
      bc_l_lo_(FaceBC::FromConfig(FaceBC::Face::L_LO, cfg.bc_l_lo)),
      bc_l_hi_(FaceBC::FromConfig(FaceBC::Face::L_HI, cfg.bc_l_hi)),
      bc_m_lo_(FaceBC::FromConfig(FaceBC::Face::M_LO, cfg.bc_m_lo)),
      bc_m_hi_(FaceBC::FromConfig(FaceBC::Face::M_HI, cfg.bc_m_hi))
{}

// ============================================================
// Public entry point
// ============================================================

void Solver::Advance(double dt) {
    current_dt_ = dt;

    ExchangeAllGhosts();
    ComputeCentralUpdate();
    UpdateCentralPhysical();

    bc_l_lo_.Apply(f_, grid_, cfg_, mpi_, dt);
    bc_m_hi_.Apply(f_, grid_, cfg_, mpi_, dt);
    bc_m_lo_.Apply(f_, grid_, cfg_, mpi_, dt);
    bc_l_hi_.Apply(f_, grid_, cfg_, mpi_, dt);

    // Reconstruct only the boundary strips that the FaceBCs just wrote u for.
    // The interior was already done by UpdateCentralPhysical.  On
    // non-M-boundary ranks the m_lo/m_hi strips are no-ops because
    // UpdateCentralPhysical already covered m=1 and m=local_M there.
    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;

    if (mpi_.IsLLoBoundary()) {
        f_.UpdatePhysicalFromU(grid_, cfg_, 1, 1, 1, local_M);
    }
    if (mpi_.IsLHiBoundary()) {
        f_.UpdatePhysicalFromU(grid_, cfg_, local_L, local_L, 1, local_M);
    }
    if (mpi_.IsMLoBoundary()) {
        f_.UpdatePhysicalFromU(grid_, cfg_, 1, local_L, 1, 1);
    }
    if (mpi_.IsMHiBoundary()) {
        f_.UpdatePhysicalFromU(grid_, cfg_, 1, local_L, local_M, local_M);
    }
    f_.CopyUToU0();
}

// ============================================================
// Ghost-cell exchange (all 4 directions, all 18 arrays, one phase)
// ============================================================

void Solver::ExchangeAllGhosts() {
    // Only conservative arrays are shipped.  Physical fields are derived
    // locally from the just-received u0_* on the four ghost rings below.
    double** arrs[8] = {
        f_.u0_1.Raw(), f_.u0_2.Raw(), f_.u0_3.Raw(), f_.u0_4.Raw(),
        f_.u0_5.Raw(), f_.u0_6.Raw(), f_.u0_7.Raw(), f_.u0_8.Raw()
    };
    mpi_.ExchangeGhostsBatch(arrs, 8, col_batch_buf_);

    // Reconstruct physical fields in the four ghost rings.  Skip rings on
    // physical-domain boundaries (no neighbour, no fresh data to derive from).
    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;

    if (mpi_.nbr_l_lo != MPI_PROC_NULL) {
        f_.UpdatePhysicalFromU0(grid_, cfg_, 0, 0, 1, local_M);
    }
    if (mpi_.nbr_l_hi != MPI_PROC_NULL) {
        f_.UpdatePhysicalFromU0(grid_, cfg_, local_L + 1, local_L + 1, 1, local_M);
    }
    if (mpi_.nbr_m_lo != MPI_PROC_NULL) {
        f_.UpdatePhysicalFromU0(grid_, cfg_, 1, local_L, 0, 0);
    }
    if (mpi_.nbr_m_hi != MPI_PROC_NULL) {
        f_.UpdatePhysicalFromU0(grid_, cfg_, 1, local_L, local_M + 1, local_M + 1);
    }
}

// ============================================================
// Lax–Friedrichs central update (interior cells)
// ============================================================

void Solver::ComputeCentralUpdate() {
    const int    local_L = mpi_.local_L;
    const int    local_M = mpi_.local_M;
    const double dt      = current_dt_;
    const double dz      = cfg_.dz;

    // On M boundary ranks the first/last interior row of the M direction
    // belongs to a BC; skip it in the central update so the BC result is
    // not overwritten.
    const int m_lo = mpi_.IsMLoBoundary() ? 2 : 1;
    const int m_hi = mpi_.IsMHiBoundary() ? local_M - 1 : local_M;

    auto** u0_1 = f_.u0_1.Raw();  auto** u0_2 = f_.u0_2.Raw();
    auto** u0_3 = f_.u0_3.Raw();  auto** u0_4 = f_.u0_4.Raw();
    auto** u0_5 = f_.u0_5.Raw();  auto** u0_6 = f_.u0_6.Raw();
    auto** u0_7 = f_.u0_7.Raw();  auto** u0_8 = f_.u0_8.Raw();

    auto** u_1  = f_.u_1.Raw();   auto** u_2  = f_.u_2.Raw();
    auto** u_3  = f_.u_3.Raw();   auto** u_4  = f_.u_4.Raw();
    auto** u_5  = f_.u_5.Raw();   auto** u_6  = f_.u_6.Raw();
    auto** u_7  = f_.u_7.Raw();   auto** u_8  = f_.u_8.Raw();

    auto** rho   = f_.rho.Raw();   auto** v_z  = f_.v_z.Raw();
    auto** v_r   = f_.v_r.Raw();   auto** v_phi= f_.v_phi.Raw();
    auto** p     = f_.p.Raw();     auto** P    = f_.P.Raw();
    auto** H_z   = f_.H_z.Raw();   auto** H_r  = f_.H_r.Raw();
    auto** H_phi = f_.H_phi.Raw();
    auto** r     = grid_.r.Raw();
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

void Solver::UpdateCentralPhysical() {
    const int m_lo = mpi_.IsMLoBoundary() ? 2 : 1;
    const int m_hi = mpi_.IsMHiBoundary() ? mpi_.local_M - 1 : mpi_.local_M;
    f_.UpdatePhysicalFromU(grid_, cfg_, 1, mpi_.local_L, m_lo, m_hi);
}
