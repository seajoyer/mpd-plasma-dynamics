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
      // prev (1×1 if unused – zero overhead)
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

void Fields::UpdatePhysicalFromU(const Grid& grid, const SimConfig& cfg,
                                      int l_lo, int l_hi,
                                      int m_lo, int m_hi) {
    const double gamma = cfg.gamma;

    #pragma omp parallel for
    for (int l = l_lo; l <= l_hi; ++l) {
        for (int m = m_lo; m <= m_hi; ++m) {
            const double inv_r  = grid.inv_r[l][m];
            const double inv_u1 = 1.0 / u_1[l][m];

            rho  [l][m] = u_1[l][m] * inv_r;
            v_z  [l][m] = u_2[l][m] * inv_u1;
            v_r  [l][m] = u_3[l][m] * inv_u1;
            v_phi[l][m] = u_4[l][m] * inv_u1;

            H_phi[l][m] = u_6[l][m];
            H_z  [l][m] = u_7[l][m] * inv_r;
            H_r  [l][m] = u_8[l][m] * inv_r;

            e[l][m] = u_5[l][m] * inv_u1;
            p[l][m] = (gamma - 1.0) * rho[l][m] * e[l][m];
            P[l][m] = p[l][m] + 0.5 * (H_z[l][m]*H_z[l][m]
                                       + H_r[l][m]*H_r[l][m]
                                       + H_phi[l][m]*H_phi[l][m]);
        }
    }
}

void Fields::UpdatePhysicalFromU0(const Grid& grid, const SimConfig& cfg,
                                  int l_lo, int l_hi,
                                  int m_lo, int m_hi) {
    const double gamma = cfg.gamma;

    #pragma omp parallel for
    for (int l = l_lo; l <= l_hi; ++l) {
        for (int m = m_lo; m <= m_hi; ++m) {
            const double inv_r  = grid.inv_r[l][m];
            const double inv_u1 = 1.0 / u0_1[l][m];

            rho  [l][m] = u0_1[l][m] * inv_r;
            v_z  [l][m] = u0_2[l][m] * inv_u1;
            v_r  [l][m] = u0_3[l][m] * inv_u1;
            v_phi[l][m] = u0_4[l][m] * inv_u1;

            H_phi[l][m] = u0_6[l][m];
            H_z  [l][m] = u0_7[l][m] * inv_r;
            H_r  [l][m] = u0_8[l][m] * inv_r;

            e[l][m] = u0_5[l][m] * inv_u1;
            p[l][m] = (gamma - 1.0) * rho[l][m] * e[l][m];
            P[l][m] = p[l][m] + 0.5 * (H_z[l][m]*H_z[l][m]
                                       + H_r[l][m]*H_r[l][m]
                                       + H_phi[l][m]*H_phi[l][m]);
        }
    }
}

void Fields::CopyUToU0() {
    #pragma omp parallel for
    for (int l = 0; l < rows; ++l) {
        for (int m = 0; m < cols; ++m) {
            u0_1[l][m] = u_1[l][m];
            u0_2[l][m] = u_2[l][m];
            u0_3[l][m] = u_3[l][m];
            u0_4[l][m] = u_4[l][m];
            u0_5[l][m] = u_5[l][m];
            u0_6[l][m] = u_6[l][m];
            u0_7[l][m] = u_7[l][m];
            u0_8[l][m] = u_8[l][m];
        }
    }
}
