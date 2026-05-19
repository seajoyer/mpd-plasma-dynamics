#include "fields.hpp"

#include <cmath>
#include <cstring>

#include "iinitial_condition.hpp"

// ---- constructor -------------------------------------------------------

Fields::Fields(int r, int c, bool with_prev)
    : rows(r), cols(c), has_prev(with_prev),
      // conservative
      u0_1(r,c), u0_2(r,c), u0_3(r,c), u0_4(r,c),
      u0_5(r,c), u0_6(r,c), u0_7(r,c), u0_8(r,c),
      u_1 (r,c), u_2 (r,c), u_3 (r,c), u_4 (r,c),
      u_5 (r,c), u_6 (r,c), u_7 (r,c), u_8 (r,c),
      // physical
      rho(r,c), v_z(r,c), v_r(r,c), v_phi(r,c),
      e(r,c),   p(r,c),   P(r,c),
      H_z(r,c), H_r(r,c), H_phi(r,c),
      // prev
      rho_prev (with_prev ? r : 1, with_prev ? c : 1),
      v_z_prev (with_prev ? r : 1, with_prev ? c : 1),
      v_r_prev (with_prev ? r : 1, with_prev ? c : 1),
      v_phi_prev(with_prev ? r : 1, with_prev ? c : 1),
      H_z_prev (with_prev ? r : 1, with_prev ? c : 1),
      H_r_prev (with_prev ? r : 1, with_prev ? c : 1),
      H_phi_prev(with_prev ? r : 1, with_prev ? c : 1)
{}

// ---- initialisation ----------------------------------------------------

void Fields::InitPhysical(const IInitialCondition& ic, const SimConfig& cfg,
                          const Grid& grid, int l_start) {
    ic.Apply(*this, grid, cfg, l_start);
}

void Fields::InitConservative(const Grid& grid) {
    #pragma omp parallel for
    for (int l = 1; l < rows - 1; ++l) {
        for (int m = 1; m < cols - 1; ++m) {
            u0_1[l][m] = rho [l][m] * grid.r[l][m];
            u0_2[l][m] = rho [l][m] * v_z [l][m] * grid.r[l][m];
            u0_3[l][m] = rho [l][m] * v_r [l][m] * grid.r[l][m];
            u0_4[l][m] = rho [l][m] * v_phi[l][m] * grid.r[l][m];
            u0_5[l][m] = rho [l][m] * e   [l][m] * grid.r[l][m];
            u0_6[l][m] = H_phi[l][m];
            u0_7[l][m] = H_z  [l][m] * grid.r[l][m];
            u0_8[l][m] = H_r  [l][m] * grid.r[l][m];
        }
    }
}

// ---- per-step helpers --------------------------------------------------
void Fields::SavePrev() {
    if (!has_prev) return;

    const std::size_t nbytes = rho.Size() * sizeof(double);
    std::memcpy(rho_prev .Flat(), rho  .Flat(), nbytes);
    std::memcpy(v_z_prev .Flat(), v_z  .Flat(), nbytes);
    std::memcpy(v_r_prev .Flat(), v_r  .Flat(), nbytes);
    std::memcpy(v_phi_prev.Flat(),v_phi .Flat(), nbytes);
    std::memcpy(H_z_prev .Flat(), H_z  .Flat(), nbytes);
    std::memcpy(H_r_prev .Flat(), H_r  .Flat(), nbytes);
    std::memcpy(H_phi_prev.Flat(),H_phi .Flat(), nbytes);
}

// ----------------------------------------------------------------
// Physical-variable reconstruction kernels
// ----------------------------------------------------------------
//
// Both UpdatePhysicalFromU and UpdatePhysicalFromU0 are pure cell-local maps,
// so the inner-m loop is trivially vectorisable once the
// compiler can prove non-aliasing.  We therefore:
//   1. Hoist each Array2D row to a typed pointer outside the inner loop.
//   2. Mark all output rows __restrict__ so the compiler may assume they do
//      not alias the input rows (true here because conservative arrays
//      (u_*) and physical arrays (rho, v_*, ...) are distinct allocations).
//   3. Apply #pragma omp simd on the inner loop.
// ----------------------------------------------------------------

void Fields::UpdatePhysicalFromU(const Grid& grid, const SimConfig& cfg,
                                 int l_lo, int l_hi,
                                 int m_lo, int m_hi) {
    const double gamma_m1 = cfg.gamma - 1.0;
    constexpr int kOmpCellThreshold = 2048;
    const int total_cells = (l_hi - l_lo + 1) * (m_hi - m_lo + 1);

    const double** inv_r_arr = grid.inv_r.Raw();

    #pragma omp parallel for if(total_cells >= kOmpCellThreshold)
    for (int l = l_lo; l <= l_hi; ++l) {
        const double* __restrict__ u1_l    = u_1[l];
        const double* __restrict__ u2_l    = u_2[l];
        const double* __restrict__ u3_l    = u_3[l];
        const double* __restrict__ u4_l    = u_4[l];
        const double* __restrict__ u5_l    = u_5[l];
        const double* __restrict__ u6_l    = u_6[l];
        const double* __restrict__ u7_l    = u_7[l];
        const double* __restrict__ u8_l    = u_8[l];
        const double* __restrict__ invr_l  = inv_r_arr[l];

        double* __restrict__ rho_l   = rho[l];
        double* __restrict__ vz_l    = v_z[l];
        double* __restrict__ vr_l    = v_r[l];
        double* __restrict__ vphi_l  = v_phi[l];
        double* __restrict__ Hphi_l  = H_phi[l];
        double* __restrict__ Hz_l    = H_z[l];
        double* __restrict__ Hr_l    = H_r[l];
        double* __restrict__ e_l     = e[l];
        double* __restrict__ p_l     = p[l];
        double* __restrict__ P_l     = P[l];

        #pragma omp simd
        for (int m = m_lo; m <= m_hi; ++m) {
            const double inv_r  = invr_l[m];
            const double inv_u1 = 1.0 / u1_l[m];

            const double rho_val = u1_l[m] * inv_r;
            const double vz_val  = u2_l[m] * inv_u1;
            const double vr_val  = u3_l[m] * inv_u1;
            const double vphi_val= u4_l[m] * inv_u1;
            const double Hphi_val= u6_l[m];
            const double Hz_val  = u7_l[m] * inv_r;
            const double Hr_val  = u8_l[m] * inv_r;
            const double e_val   = u5_l[m] * inv_u1;
            const double p_val   = gamma_m1 * rho_val * e_val;

            rho_l  [m] = rho_val;
            vz_l   [m] = vz_val;
            vr_l   [m] = vr_val;
            vphi_l [m] = vphi_val;
            Hphi_l [m] = Hphi_val;
            Hz_l   [m] = Hz_val;
            Hr_l   [m] = Hr_val;
            e_l    [m] = e_val;
            p_l    [m] = p_val;
            P_l    [m] = p_val + 0.5 * (Hz_val*Hz_val
                                        + Hr_val*Hr_val
                                        + Hphi_val*Hphi_val);
        }
    }
}

void Fields::UpdatePhysicalFromU0(const Grid& grid, const SimConfig& cfg,
                                  int l_lo, int l_hi,
                                  int m_lo, int m_hi) {
    const double gamma_m1 = cfg.gamma - 1.0;
    constexpr int kOmpCellThreshold = 2048;
    const int total_cells = (l_hi - l_lo + 1) * (m_hi - m_lo + 1);

    const double** inv_r_arr = grid.inv_r.Raw();

    #pragma omp parallel for if(total_cells >= kOmpCellThreshold)
    for (int l = l_lo; l <= l_hi; ++l) {
        const double* __restrict__ u01_l   = u0_1[l];
        const double* __restrict__ u02_l   = u0_2[l];
        const double* __restrict__ u03_l   = u0_3[l];
        const double* __restrict__ u04_l   = u0_4[l];
        const double* __restrict__ u05_l   = u0_5[l];
        const double* __restrict__ u06_l   = u0_6[l];
        const double* __restrict__ u07_l   = u0_7[l];
        const double* __restrict__ u08_l   = u0_8[l];
        const double* __restrict__ invr_l  = inv_r_arr[l];

        double* __restrict__ rho_l   = rho[l];
        double* __restrict__ vz_l    = v_z[l];
        double* __restrict__ vr_l    = v_r[l];
        double* __restrict__ vphi_l  = v_phi[l];
        double* __restrict__ Hphi_l  = H_phi[l];
        double* __restrict__ Hz_l    = H_z[l];
        double* __restrict__ Hr_l    = H_r[l];
        double* __restrict__ e_l     = e[l];
        double* __restrict__ p_l     = p[l];
        double* __restrict__ P_l     = P[l];

        #pragma omp simd
        for (int m = m_lo; m <= m_hi; ++m) {
            const double inv_r  = invr_l[m];
            const double inv_u1 = 1.0 / u01_l[m];

            const double rho_val = u01_l[m] * inv_r;
            const double vz_val  = u02_l[m] * inv_u1;
            const double vr_val  = u03_l[m] * inv_u1;
            const double vphi_val= u04_l[m] * inv_u1;
            const double Hphi_val= u06_l[m];
            const double Hz_val  = u07_l[m] * inv_r;
            const double Hr_val  = u08_l[m] * inv_r;
            const double e_val   = u05_l[m] * inv_u1;
            const double p_val   = gamma_m1 * rho_val * e_val;

            rho_l  [m] = rho_val;
            vz_l   [m] = vz_val;
            vr_l   [m] = vr_val;
            vphi_l [m] = vphi_val;
            Hphi_l [m] = Hphi_val;
            Hz_l   [m] = Hz_val;
            Hr_l   [m] = Hr_val;
            e_l    [m] = e_val;
            p_l    [m] = p_val;
            P_l    [m] = p_val + 0.5 * (Hz_val*Hz_val
                                        + Hr_val*Hr_val
                                        + Hphi_val*Hphi_val);
        }
    }
}
