#include "solver.hpp"

#include <cmath>
#include <utility>

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

    std::swap(f_.u_1, f_.u0_1);
    std::swap(f_.u_2, f_.u0_2);
    std::swap(f_.u_3, f_.u0_3);
    std::swap(f_.u_4, f_.u0_4);
    std::swap(f_.u_5, f_.u0_5);
    std::swap(f_.u_6, f_.u0_6);
    std::swap(f_.u_7, f_.u0_7);
    std::swap(f_.u_8, f_.u0_8);
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

    // Raw double** pointers — one dereference per row access inside the loop.
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
    const double dt_inv_2dz = dt / (2.0 * dz);

    // Parallelise on l
    #pragma omp parallel for
    for (int l = 1; l <= local_L; ++l) {
        const double dt_inv_2drl = dt / (2.0 * dr[l]);

        // ---- Hoist row pointers for all three l-rows used in the stencil ----
        // Conservative state (three rows each: l-1, l, l+1)
        const double* u0_1_lm = u0_1[l-1]; const double* u0_1_l = u0_1[l]; const double* u0_1_lp = u0_1[l+1];
        const double* u0_2_lm = u0_2[l-1]; const double* u0_2_l = u0_2[l]; const double* u0_2_lp = u0_2[l+1];
        const double* u0_3_lm = u0_3[l-1]; const double* u0_3_l = u0_3[l]; const double* u0_3_lp = u0_3[l+1];
        const double* u0_4_lm = u0_4[l-1]; const double* u0_4_l = u0_4[l]; const double* u0_4_lp = u0_4[l+1];
        const double* u0_5_lm = u0_5[l-1]; const double* u0_5_l = u0_5[l]; const double* u0_5_lp = u0_5[l+1];
        const double* u0_6_lm = u0_6[l-1]; const double* u0_6_l = u0_6[l]; const double* u0_6_lp = u0_6[l+1];
        const double* u0_7_lm = u0_7[l-1]; const double* u0_7_l = u0_7[l]; const double* u0_7_lp = u0_7[l+1];
        const double* u0_8_lm = u0_8[l-1]; const double* u0_8_l = u0_8[l]; const double* u0_8_lp = u0_8[l+1];

        // Output rows
        double* u_1_l  = u_1[l];  double* u_2_l  = u_2[l];
        double* u_3_l  = u_3[l];  double* u_4_l  = u_4[l];
        double* u_5_l  = u_5[l];  double* u_6_l  = u_6[l];
        double* u_7_l  = u_7[l];  double* u_8_l  = u_8[l];

        // Physical fields (three rows where needed)
        const double* vz_lm  = v_z[l-1];   const double* vz_l  = v_z[l];   const double* vz_lp  = v_z[l+1];
        const double* vr_lm  = v_r[l-1];                                   const double* vr_lp  = v_r[l+1];
        const double* vphi_lm= v_phi[l-1];                                 const double* vphi_lp= v_phi[l+1];
        const double* Hz_lm  = H_z[l-1];   const double* Hz_l  = H_z[l];   const double* Hz_lp  = H_z[l+1];
        const double* Hr_lm  = H_r[l-1];                                   const double* Hr_lp  = H_r[l+1];
        const double* Hphi_lm= H_phi[l-1]; const double* Hphi_l= H_phi[l]; const double* Hphi_lp= H_phi[l+1];
        const double* P_lm   = P[l-1];                                     const double* P_lp   = P[l+1];
        const double* p_l    = p[l];                                       
        const double* r_lm   = r[l-1];     const double* r_l   = r[l];     const double* r_lp   = r[l+1];
        const double* rho_l  = rho[l];
        const double* vphi_l = v_phi[l];
        const double* Hphi_l2= H_phi[l]; // alias for same row, separate name for clarity

        for (int m = m_lo; m <= m_hi; ++m) {
            u_1_l[m] =
                0.25 * (u0_1_lp[m] + u0_1_lm[m] + u0_1_l[m+1] + u0_1_l[m-1])
                - dt_inv_2dz  * (u0_1_lp[m]*vz_lp[m]   - u0_1_lm[m]*vz_lm[m])
                - dt_inv_2drl * (u0_1_l[m+1]*vr_lp[m+1] - u0_1_l[m-1]*vr_lp[m-1]);

            u_2_l[m] =
                0.25 * (u0_2_lp[m] + u0_2_lm[m] + u0_2_l[m+1] + u0_2_l[m-1])
                + dt_inv_2dz  * ( (Hz_lp[m]*Hz_lp[m] - P_lp[m])*r_lp[m]
                                 -(Hz_lm[m]*Hz_lm[m] - P_lm[m])*r_lm[m])
                + dt_inv_2drl * ( Hz_l[m+1]*Hr_lp[m+1]*r_l[m+1]
                                 -Hz_l[m-1]*Hr_lp[m-1]*r_l[m-1])
                - dt_inv_2dz  * (u0_2_lp[m]*vz_lp[m]   - u0_2_lm[m]*vz_lm[m])
                - dt_inv_2drl * (u0_2_l[m+1]*vr_lp[m+1] - u0_2_l[m-1]*vr_lp[m-1]);

            u_3_l[m] =
                0.25 * (u0_3_lp[m] + u0_3_lm[m] + u0_3_l[m+1] + u0_3_l[m-1])
                + dt * (rho_l[m]*vphi_l[m]*vphi_l[m] + P_lp[m] - Hphi_l2[m]*Hphi_l2[m])
                + dt_inv_2dz  * ( Hz_lp[m]*Hr_lp[m]*r_lp[m]
                                 -Hz_lm[m]*Hr_lm[m]*r_lm[m])
                + dt_inv_2drl * ( (Hr_lp[m+1]*Hr_lp[m+1] - P_lp[m+1])*r_l[m+1]
                                 -(Hr_lp[m-1]*Hr_lp[m-1] - P_lp[m-1])*r_l[m-1])
                - dt_inv_2dz  * (u0_3_lp[m]*vz_lp[m]   - u0_3_lm[m]*vz_lm[m])
                - dt_inv_2drl * (u0_3_l[m+1]*vr_lp[m+1] - u0_3_l[m-1]*vr_lp[m-1]);

            u_4_l[m] =
                0.25 * (u0_4_lp[m] + u0_4_lm[m] + u0_4_l[m+1] + u0_4_l[m-1])
                + dt * (-rho_l[m]*v_r[l][m]*vphi_l[m] + Hphi_l2[m]*Hr_lp[m])
                + dt_inv_2dz  * ( Hphi_lp[m]*Hz_lp[m]*r_lp[m]
                                 -Hphi_lm[m]*Hz_lm[m]*r_lm[m])
                + dt_inv_2drl * ( Hphi_l[m+1]*Hr_lp[m+1]*r_l[m+1]
                                 -Hphi_l[m-1]*Hr_lp[m-1]*r_l[m-1])
                - dt_inv_2dz  * (u0_4_lp[m]*vz_lp[m]   - u0_4_lm[m]*vz_lm[m])
                - dt_inv_2drl * (u0_4_l[m+1]*vr_lp[m+1] - u0_4_l[m-1]*vr_lp[m-1]);

            u_5_l[m] =
                0.25 * (u0_5_lp[m] + u0_5_lm[m] + u0_5_l[m+1] + u0_5_l[m-1])
                - p_l[m] * (dt_inv_2dz  * (vz_lp[m]*r_lp[m]   - vz_lm[m]*r_lm[m])
                           + dt_inv_2drl * (v_r[l][m+1]*r_l[m+1] - v_r[l][m-1]*r_l[m-1]))
                - dt_inv_2dz  * (u0_5_lp[m]*vz_lp[m]   - u0_5_lm[m]*vz_lm[m])
                - dt_inv_2drl * (u0_5_l[m+1]*vr_lp[m+1] - u0_5_l[m-1]*vr_lp[m-1]);

            u_6_l[m] =
                0.25 * (u0_6_lp[m] + u0_6_lm[m] + u0_6_l[m+1] + u0_6_l[m-1])
                + dt_inv_2dz  * (Hz_lp[m]*vphi_lp[m]   - Hz_lm[m]*vphi_lm[m])
                + dt_inv_2drl * (Hr_lp[m+1]*vphi_l[m+1] - Hr_lp[m-1]*vphi_l[m-1])
                - dt_inv_2dz  * (u0_6_lp[m]*vz_lp[m]   - u0_6_lm[m]*vz_lm[m])
                - dt_inv_2drl * (u0_6_l[m+1]*vr_lp[m+1] - u0_6_l[m-1]*vr_lp[m-1]);

            u_7_l[m] =
                0.25 * (u0_7_lp[m] + u0_7_lm[m] + u0_7_l[m+1] + u0_7_l[m-1])
                + dt_inv_2drl * (Hr_lp[m+1]*vz_l[m+1]*r_l[m+1] - Hr_lp[m-1]*vz_l[m-1]*r_l[m-1])
                - dt_inv_2drl * (u0_7_l[m+1]*vr_lp[m+1] - u0_7_l[m-1]*vr_lp[m-1]);

            u_8_l[m] =
                0.25 * (u0_8_lp[m] + u0_8_lm[m] + u0_8_l[m+1] + u0_8_l[m-1])
                + dt_inv_2dz * (Hz_lp[m]*v_r[l+1][m]*r_lp[m] - Hz_lm[m]*v_r[l-1][m]*r_lm[m])
                - dt_inv_2dz * (u0_8_lp[m]*vz_lp[m] - u0_8_lm[m]*vz_lm[m]);
        }
    }
}

void Solver::UpdateCentralPhysical() {
    const int m_lo = mpi_.IsMLoBoundary() ? 2 : 1;
    const int m_hi = mpi_.IsMHiBoundary() ? mpi_.local_M - 1 : mpi_.local_M;
    f_.UpdatePhysicalFromU(grid_, cfg_, 1, mpi_.local_L, m_lo, m_hi);
}
