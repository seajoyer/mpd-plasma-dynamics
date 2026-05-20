#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

// ============================================================
// Constructor — build FaceBC objects from config, cache
// invariant grid pointers and per-rank LF boundary offsets.
// ============================================================

Solver::Solver(const SimConfig& cfg, const MPIManager& mpi,
               const Grid& grid, Fields& f)
    : cfg_(cfg), mpi_(mpi), grid_(grid), f_(f),
      bc_l_lo_(FaceBC::FromConfig(FaceBC::Face::L_LO, cfg.bc_l_lo)),
      bc_l_hi_(FaceBC::FromConfig(FaceBC::Face::L_HI, cfg.bc_l_hi)),
      bc_m_lo_(FaceBC::FromConfig(FaceBC::Face::M_LO, cfg.bc_m_lo)),
      bc_m_hi_(FaceBC::FromConfig(FaceBC::Face::M_HI, cfg.bc_m_hi))
{
    // The LF kernel skips the boundary strips on M-boundary ranks because the
    // M BCs will overwrite them.  These offsets are determined by the MPI
    // decomposition (fixed at start-up) so we compute them once and cache.
    m_lo_bc_ = mpi_.IsMLoBoundary() ? 2 : 1;
    m_hi_bc_ = mpi_.IsMHiBoundary() ? mpi_.local_M - 1 : mpi_.local_M;

    // Grid arrays never change after Grid::Build — bind their raw pointers
    // once so the per-step kernel doesn't pay for Raw()/data() each call.
    r_ptr_  = grid_.r.Raw();
    dr_ptr_ = grid_.dr.data();
}

// ============================================================
// Public entry point — with comm/compute overlap
// ============================================================

void Solver::Advance(double dt) {
    current_dt_ = dt;

    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;
    const int m_lo_bc = m_lo_bc_;
    const int m_hi_bc = m_hi_bc_;

    // Deep-interior bounds — cells whose 5-point stencil cannot reach any
    // ghost ring (l=0, l=local_L+1, m=0, m=local_M+1).  These can be updated
    // while the ghost exchange is still in flight.
    const int l_lo_inner = 2;
    const int l_hi_inner = local_L - 1;
    const int m_lo_inner = std::max(m_lo_bc, 2);
    const int m_hi_inner = std::min(m_hi_bc, local_M - 1);

    // ---- 1. Post non-blocking ghost exchanges -----------------------------
    PostGhostExchange();

    // ---- 2. LF update on the deep interior --------------------------------
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
// Lax–Friedrichs central update
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
    const double** r  = r_ptr_;   // cached once in ctor
    const double*  dr = dr_ptr_;  // cached once in ctor
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

        double* __restrict__ u_1_l = u_1[l];  double* __restrict__ u_2_l = u_2[l];
        double* __restrict__ u_3_l = u_3[l];  double* __restrict__ u_4_l = u_4[l];
        double* __restrict__ u_5_l = u_5[l];  double* __restrict__ u_6_l = u_6[l];
        double* __restrict__ u_7_l = u_7[l];  double* __restrict__ u_8_l = u_8[l];

        const double* vz_lm  = v_z[l-1];   const double* vz_l  = v_z[l];   const double* vz_lp  = v_z[l+1];
        const double* vr_lm  = v_r[l-1];   const double* vr_l  = v_r[l];   const double* vr_lp  = v_r[l+1];
        const double* vphi_lm= v_phi[l-1]; const double* vphi_l= v_phi[l]; const double* vphi_lp= v_phi[l+1];
        const double* Hz_lm  = H_z[l-1];   const double* Hz_l  = H_z[l];   const double* Hz_lp  = H_z[l+1];
        const double* Hr_lm  = H_r[l-1];   const double* Hr_l  = H_r[l];   const double* Hr_lp  = H_r[l+1];
        const double* Hphi_lm= H_phi[l-1]; const double* Hphi_l= H_phi[l]; const double* Hphi_lp= H_phi[l+1];
        const double* P_lm   = P[l-1];     const double* P_l   = P[l];     const double* P_lp   = P[l+1];
        const double* p_l    = p[l];
        const double* r_lm   = r[l-1];     const double* r_l   = r[l];     const double* r_lp   = r[l+1];
        const double* rho_l  = rho[l];

        #pragma omp simd
        for (int m = m_lo; m <= m_hi; ++m) {
            // ── L-axis composites at l±1 (single cell, no carry) ───────
            const double r_lp_m    = r_lp[m];
            const double r_lm_m    = r_lm[m];
            const double Hz_lp_m   = Hz_lp[m];
            const double Hz_lm_m   = Hz_lm[m];
            const double Hr_lp_m   = Hr_lp[m];
            const double Hr_lm_m   = Hr_lm[m];
            const double Hphi_lp_m = Hphi_lp[m];
            const double Hphi_lm_m = Hphi_lm[m];
            const double vz_lp_m   = vz_lp[m];
            const double vz_lm_m   = vz_lm[m];
            const double vr_lp_m   = vr_lp[m];
            const double vr_lm_m   = vr_lm[m];
            const double vphi_lp_m = vphi_lp[m];
            const double vphi_lm_m = vphi_lm[m];
            const double P_lp_m    = P_lp[m];
            const double P_lm_m    = P_lm[m];

            // r·Hz at l±1  (reused 4× per cell: u_2, u_3, u_4, u_8 z-fluxes)
            const double rHz_lp = r_lp_m * Hz_lp_m;
            const double rHz_lm = r_lm_m * Hz_lm_m;

            // (Hz² − P)·r at l±1  (u_2)
            const double Hz2mP_r_lp = rHz_lp * Hz_lp_m - P_lp_m * r_lp_m;
            const double Hz2mP_r_lm = rHz_lm * Hz_lm_m - P_lm_m * r_lm_m;

            // Hz·Hr·r at l±1  (u_3 z-flux)
            const double HzHr_r_lp = rHz_lp * Hr_lp_m;
            const double HzHr_r_lm = rHz_lm * Hr_lm_m;

            // Hphi·Hz·r at l±1  (u_4 z-flux)
            const double HphiHz_r_lp = rHz_lp * Hphi_lp_m;
            const double HphiHz_r_lm = rHz_lm * Hphi_lm_m;

            // Hz·vr·r at l±1  (u_8 z-flux)
            const double Hzvr_r_lp = rHz_lp * vr_lp_m;
            const double Hzvr_r_lm = rHz_lm * vr_lm_m;

            // vz·r at l±1  (u_5 energy)
            const double vz_r_lp = vz_lp_m * r_lp_m;
            const double vz_r_lm = vz_lm_m * r_lm_m;

            // Hz·vphi at l±1  (u_6 z-flux, no r factor)
            const double Hzvphi_lp = Hz_lp_m * vphi_lp_m;
            const double Hzvphi_lm = Hz_lm_m * vphi_lm_m;

            // Advective u0_k·vz at l±1  (k = 1..6, 8; u_7 has no z-flux)
            const double u0vz_1_lp = u0_1_lp[m] * vz_lp_m;
            const double u0vz_1_lm = u0_1_lm[m] * vz_lm_m;
            const double u0vz_2_lp = u0_2_lp[m] * vz_lp_m;
            const double u0vz_2_lm = u0_2_lm[m] * vz_lm_m;
            const double u0vz_3_lp = u0_3_lp[m] * vz_lp_m;
            const double u0vz_3_lm = u0_3_lm[m] * vz_lm_m;
            const double u0vz_4_lp = u0_4_lp[m] * vz_lp_m;
            const double u0vz_4_lm = u0_4_lm[m] * vz_lm_m;
            const double u0vz_5_lp = u0_5_lp[m] * vz_lp_m;
            const double u0vz_5_lm = u0_5_lm[m] * vz_lm_m;
            const double u0vz_6_lp = u0_6_lp[m] * vz_lp_m;
            const double u0vz_6_lm = u0_6_lm[m] * vz_lm_m;
            const double u0vz_8_lp = u0_8_lp[m] * vz_lp_m;
            const double u0vz_8_lm = u0_8_lm[m] * vz_lm_m;

            // ── M-axis composites at m±1 (single cell, no carry) ───────
            const double r_mp    = r_l[m+1];
            const double r_mn    = r_l[m-1];
            const double Hr_mp   = Hr_l[m+1];
            const double Hr_mn   = Hr_l[m-1];
            const double Hz_mp   = Hz_l[m+1];
            const double Hz_mn   = Hz_l[m-1];
            const double Hphi_mp = Hphi_l[m+1];
            const double Hphi_mn = Hphi_l[m-1];
            const double vz_mp   = vz_l[m+1];
            const double vz_mn   = vz_l[m-1];
            const double vr_mp   = vr_l[m+1];
            const double vr_mn   = vr_l[m-1];
            const double vphi_mp = vphi_l[m+1];
            const double vphi_mn = vphi_l[m-1];
            const double P_mp    = P_l[m+1];
            const double P_mn    = P_l[m-1];

            // r·Hr at m±1  (reused 4× per cell: u_2, u_3, u_4, u_7 r-fluxes)
            const double rHr_mp = r_mp * Hr_mp;
            const double rHr_mn = r_mn * Hr_mn;

            // (Hr² − P)·r at m±1  (u_3 r-flux)
            const double Hr2mP_r_mp = rHr_mp * Hr_mp - P_mp * r_mp;
            const double Hr2mP_r_mn = rHr_mn * Hr_mn - P_mn * r_mn;

            // Hz·Hr·r at m±1  (u_2 r-flux)
            const double HzHr_r_mp = rHr_mp * Hz_mp;
            const double HzHr_r_mn = rHr_mn * Hz_mn;

            // Hphi·Hr·r at m±1  (u_4 r-flux)
            const double HphiHr_r_mp = rHr_mp * Hphi_mp;
            const double HphiHr_r_mn = rHr_mn * Hphi_mn;

            // Hr·vz·r at m±1  (u_7 r-flux)
            const double Hrvz_r_mp = rHr_mp * vz_mp;
            const double Hrvz_r_mn = rHr_mn * vz_mn;

            // vr·r at m±1  (u_5 energy)
            const double vr_r_mp = vr_mp * r_mp;
            const double vr_r_mn = vr_mn * r_mn;

            // Hr·vphi at m±1  (u_6 r-flux, no r factor)
            const double Hrvphi_mp = Hr_mp * vphi_mp;
            const double Hrvphi_mn = Hr_mn * vphi_mn;

            // Advective u0_k·vr at m±1  (k = 1..8)
            const double u0vr_1_mp = u0_1_l[m+1] * vr_mp;
            const double u0vr_1_mn = u0_1_l[m-1] * vr_mn;
            const double u0vr_2_mp = u0_2_l[m+1] * vr_mp;
            const double u0vr_2_mn = u0_2_l[m-1] * vr_mn;
            const double u0vr_3_mp = u0_3_l[m+1] * vr_mp;
            const double u0vr_3_mn = u0_3_l[m-1] * vr_mn;
            const double u0vr_4_mp = u0_4_l[m+1] * vr_mp;
            const double u0vr_4_mn = u0_4_l[m-1] * vr_mn;
            const double u0vr_5_mp = u0_5_l[m+1] * vr_mp;
            const double u0vr_5_mn = u0_5_l[m-1] * vr_mn;
            const double u0vr_6_mp = u0_6_l[m+1] * vr_mp;
            const double u0vr_6_mn = u0_6_l[m-1] * vr_mn;
            const double u0vr_7_mp = u0_7_l[m+1] * vr_mp;
            const double u0vr_7_mn = u0_7_l[m-1] * vr_mn;
            const double u0vr_8_mp = u0_8_l[m+1] * vr_mp;
            const double u0vr_8_mn = u0_8_l[m-1] * vr_mn;

            // ── u_1 : ρ·r ────────────────────────────────────────────────
            u_1_l[m] =
                0.25 * (u0_1_lp[m] + u0_1_lm[m] + u0_1_l[m+1] + u0_1_l[m-1])
                - dt_inv_2dz  * (u0vz_1_lp - u0vz_1_lm)
                - dt_inv_2drl * (u0vr_1_mp - u0vr_1_mn);

            // ── u_2 : ρ·v_z·r  (z-momentum) ─────────────────────────────
            u_2_l[m] =
                0.25 * (u0_2_lp[m] + u0_2_lm[m] + u0_2_l[m+1] + u0_2_l[m-1])
                + dt_inv_2dz  * (Hz2mP_r_lp - Hz2mP_r_lm)
                + dt_inv_2drl * (HzHr_r_mp  - HzHr_r_mn)
                - dt_inv_2dz  * (u0vz_2_lp  - u0vz_2_lm)
                - dt_inv_2drl * (u0vr_2_mp  - u0vr_2_mn);

            // ── u_3 : ρ·v_r·r  (r-momentum) ─────────────────────────────
            u_3_l[m] =
                0.25 * (u0_3_lp[m] + u0_3_lm[m] + u0_3_l[m+1] + u0_3_l[m-1])
                + dt * (rho_l[m]*vphi_l[m]*vphi_l[m] + P_l[m] - Hphi_l[m]*Hphi_l[m])
                + dt_inv_2dz  * (HzHr_r_lp  - HzHr_r_lm)
                + dt_inv_2drl * (Hr2mP_r_mp - Hr2mP_r_mn)
                - dt_inv_2dz  * (u0vz_3_lp  - u0vz_3_lm)
                - dt_inv_2drl * (u0vr_3_mp  - u0vr_3_mn);

            // ── u_4 : ρ·v_φ·r  (φ-momentum) ─────────────────────────────
            u_4_l[m] =
                0.25 * (u0_4_lp[m] + u0_4_lm[m] + u0_4_l[m+1] + u0_4_l[m-1])
                + dt * (-rho_l[m]*vr_l[m]*vphi_l[m] + Hphi_l[m]*Hr_l[m])
                + dt_inv_2dz  * (HphiHz_r_lp - HphiHz_r_lm)
                + dt_inv_2drl * (HphiHr_r_mp - HphiHr_r_mn)
                - dt_inv_2dz  * (u0vz_4_lp   - u0vz_4_lm)
                - dt_inv_2drl * (u0vr_4_mp   - u0vr_4_mn);

            // ── u_5 : ρ·e·r  (energy) ────────────────────────────────────
            u_5_l[m] =
                0.25 * (u0_5_lp[m] + u0_5_lm[m] + u0_5_l[m+1] + u0_5_l[m-1])
                - p_l[m] * (dt_inv_2dz  * (vz_r_lp - vz_r_lm)
                          + dt_inv_2drl * (vr_r_mp - vr_r_mn))
                - dt_inv_2dz  * (u0vz_5_lp - u0vz_5_lm)
                - dt_inv_2drl * (u0vr_5_mp - u0vr_5_mn);

            // ── u_6 : H_φ ───────────────────────────────────────────────
            u_6_l[m] =
                0.25 * (u0_6_lp[m] + u0_6_lm[m] + u0_6_l[m+1] + u0_6_l[m-1])
                + dt_inv_2dz  * (Hzvphi_lp - Hzvphi_lm)
                + dt_inv_2drl * (Hrvphi_mp - Hrvphi_mn)
                - dt_inv_2dz  * (u0vz_6_lp - u0vz_6_lm)
                - dt_inv_2drl * (u0vr_6_mp - u0vr_6_mn);

            // ── u_7 : H_z·r ─────────────────────────────────────────────
            u_7_l[m] =
                0.25 * (u0_7_lp[m] + u0_7_lm[m] + u0_7_l[m+1] + u0_7_l[m-1])
                + dt_inv_2drl * (Hrvz_r_mp - Hrvz_r_mn)
                - dt_inv_2drl * (u0vr_7_mp - u0vr_7_mn);

            // ── u_8 : H_r·r ─────────────────────────────────────────────
            u_8_l[m] =
                0.25 * (u0_8_lp[m] + u0_8_lm[m] + u0_8_l[m+1] + u0_8_l[m-1])
                + dt_inv_2dz * (Hzvr_r_lp - Hzvr_r_lm)
                - dt_inv_2dz * (u0vz_8_lp - u0vz_8_lm);
        }
    }
}

void Solver::UpdateCentralPhysical() {
    // m_lo / m_hi are MPI-decomposition invariants — cached once in the
    // constructor as m_lo_bc_ / m_hi_bc_.
    f_.UpdatePhysicalFromU(grid_, cfg_, 1, mpi_.local_L, m_lo_bc_, m_hi_bc_);
}
