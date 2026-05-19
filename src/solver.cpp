#include "solver.hpp"

#include <algorithm>
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
// Public entry point — with comm/compute overlap
// ============================================================

void Solver::Advance(double dt) {
    current_dt_ = dt;

    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;
    const int m_lo_bc = mpi_.IsMLoBoundary() ? 2 : 1;
    const int m_hi_bc = mpi_.IsMHiBoundary() ? local_M - 1 : local_M;

    // Deep-interior bounds — cells whose 5-point stencil cannot reach any
    // ghost ring (l=0, l=local_L+1, m=0, m=local_M+1).  These can be updated
    // while the ghost exchange is still in flight.
    const int l_lo_inner = 2;
    const int l_hi_inner = local_L - 1;
    const int m_lo_inner = std::max(m_lo_bc, 2);
    const int m_hi_inner = std::min(m_hi_bc, local_M - 1);

    // ---- 1. Post non-blocking ghost exchanges -----------------------------
    PostGhostExchange();

    // ---- 2. LF update on the deep interior (no ghost dependencies) --------
    if (l_lo_inner <= l_hi_inner && m_lo_inner <= m_hi_inner) {
        ComputeCentralUpdateRange(l_lo_inner, l_hi_inner, m_lo_inner, m_hi_inner);
    }

    // ---- 3. Wait for exchange + reconstruct ghost-ring physical fields ----
    FinishGhostExchange();

    // ---- 4. LF update on the boundary strips that depend on ghost rings ---
    // L-direction strips (full m range).  On L boundary ranks the LF result
    // at l=1 / l=local_L is overwritten by the corresponding L BC, so we
    // don't need to special-case those ranks.
    if (m_lo_bc <= m_hi_bc) {
        if (local_L >= 1) {
            ComputeCentralUpdateRange(1, 1, m_lo_bc, m_hi_bc);
        }
        if (local_L >= 2) {
            ComputeCentralUpdateRange(local_L, local_L, m_lo_bc, m_hi_bc);
        }
    }
    // M-direction strips on inner-l rows (corners already handled by L strips).
    // On M boundary ranks these strips are out of the LF range and skipped.
    if (l_lo_inner <= l_hi_inner) {
        if (!mpi_.IsMLoBoundary()) {
            ComputeCentralUpdateRange(l_lo_inner, l_hi_inner, 1, 1);
        }
        if (!mpi_.IsMHiBoundary()) {
            ComputeCentralUpdateRange(l_lo_inner, l_hi_inner, local_M, local_M);
        }
    }

    // ---- 5. Reconstruct physical fields for entire interior ---------------
    UpdateCentralPhysical();

    // ---- 6. Boundary conditions -------------------------------------------
    bc_l_lo_.Apply(f_, grid_, cfg_, mpi_, dt);
    bc_m_hi_.Apply(f_, grid_, cfg_, mpi_, dt);
    bc_m_lo_.Apply(f_, grid_, cfg_, mpi_, dt);
    bc_l_hi_.Apply(f_, grid_, cfg_, mpi_, dt);

    // Reconstruct only the boundary strips that the FaceBCs just wrote u for.
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

    // ---- 7. Advance u0 ← u (pointer swap, O(1)) ---------------------------
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
// Ghost-cell exchange — split into post + finish
// ============================================================

void Solver::PostGhostExchange() {
    double** arrs[8] = {
        f_.u0_1.Raw(), f_.u0_2.Raw(), f_.u0_3.Raw(), f_.u0_4.Raw(),
        f_.u0_5.Raw(), f_.u0_6.Raw(), f_.u0_7.Raw(), f_.u0_8.Raw()
    };
    mpi_.PostGhostsBatch(arrs, 8, col_batch_buf_, ghost_handle_);
}

void Solver::FinishGhostExchange() {
    double** arrs[8] = {
        f_.u0_1.Raw(), f_.u0_2.Raw(), f_.u0_3.Raw(), f_.u0_4.Raw(),
        f_.u0_5.Raw(), f_.u0_6.Raw(), f_.u0_7.Raw(), f_.u0_8.Raw()
    };
    mpi_.WaitAndUnpackGhostsBatch(arrs, 8, col_batch_buf_, ghost_handle_);

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
// Lax–Friedrichs central update — over an arbitrary (l, m) range
// ============================================================

void Solver::ComputeCentralUpdateRange(int l_lo, int l_hi, int m_lo, int m_hi) {
    if (l_lo > l_hi || m_lo > m_hi) return;

    const double dt = current_dt_;
    const double dz = cfg_.dz;

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

    constexpr int kOmpCellThreshold = 2048;
    const int total_cells = (l_hi - l_lo + 1) * (m_hi - m_lo + 1);

    #pragma omp parallel for if(total_cells >= kOmpCellThreshold)
    for (int l = l_lo; l <= l_hi; ++l) {
        const double dt_inv_2drl = dt / (2.0 * dr[l]);

        const double* u0_1_lm = u0_1[l-1]; const double* u0_1_l = u0_1[l]; const double* u0_1_lp = u0_1[l+1];
        const double* u0_2_lm = u0_2[l-1]; const double* u0_2_l = u0_2[l]; const double* u0_2_lp = u0_2[l+1];
        const double* u0_3_lm = u0_3[l-1]; const double* u0_3_l = u0_3[l]; const double* u0_3_lp = u0_3[l+1];
        const double* u0_4_lm = u0_4[l-1]; const double* u0_4_l = u0_4[l]; const double* u0_4_lp = u0_4[l+1];
        const double* u0_5_lm = u0_5[l-1]; const double* u0_5_l = u0_5[l]; const double* u0_5_lp = u0_5[l+1];
        const double* u0_6_lm = u0_6[l-1]; const double* u0_6_l = u0_6[l]; const double* u0_6_lp = u0_6[l+1];
        const double* u0_7_lm = u0_7[l-1]; const double* u0_7_l = u0_7[l]; const double* u0_7_lp = u0_7[l+1];
        const double* u0_8_lm = u0_8[l-1]; const double* u0_8_l = u0_8[l]; const double* u0_8_lp = u0_8[l+1];

        double* u_1_l  = u_1[l];  double* u_2_l  = u_2[l];
        double* u_3_l  = u_3[l];  double* u_4_l  = u_4[l];
        double* u_5_l  = u_5[l];  double* u_6_l  = u_6[l];
        double* u_7_l  = u_7[l];  double* u_8_l  = u_8[l];

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
        const double* Hphi_l2= H_phi[l];

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
