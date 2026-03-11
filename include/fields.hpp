#pragma once

#include "array2d.hpp"
#include "config.hpp"
#include "grid.hpp"

/// Owns every 2-D field array needed by the simulation.
///
/// Conservative variable naming follows the original code:
///   u0_* = values at the beginning of the current time step
///   u_*  = values being computed for the next time step
///
/// Physical variable naming is self-explanatory (rho, v_z, …).
///
/// Optional "prev" arrays (rho_prev, …) are allocated only when
/// convergence checking is enabled (has_prev == true).
class Fields {
public:
    int rows;   ///< local_L_with_ghosts
    int cols;   ///< M_max + 1
    bool has_prev;

    // ---- conservative variables ----
    Array2D u0_1, u0_2, u0_3, u0_4, u0_5, u0_6, u0_7, u0_8;
    Array2D u_1,  u_2,  u_3,  u_4,  u_5,  u_6,  u_7,  u_8;

    // ---- physical variables ----
    Array2D rho, v_z, v_r, v_phi;
    Array2D e, p, P;
    Array2D H_z, H_r, H_phi;

    // ---- previous-step snapshots for convergence checking ----
    Array2D rho_prev,   v_z_prev,   v_r_prev,   v_phi_prev;
    Array2D H_z_prev,   H_r_prev,   H_phi_prev;

    /// @param rows      local_L_with_ghosts
    /// @param cols      M_max + 1
    /// @param with_prev allocate prev arrays
    Fields(int rows, int cols, bool with_prev = false);

    // ---- initialisation ----

    /// Set initial conditions on physical arrays for interior cells.
    void init_physical(const SimConfig& cfg, const Grid& grid, int l_start);

    /// Compute conservative arrays u0_* from the already-set physical arrays.
    void init_conservative(const Grid& grid);

    // ---- per-step helpers ----

    /// Copy current physical arrays into the prev snapshot.
    void save_prev();

    /// Update physical vars from u_* for a range of l-indices (inclusive).
    /// m range: [m_lo, m_hi] inclusive.
    void update_physical_from_u(const Grid& grid, const SimConfig& cfg,
                                 int l_lo, int l_hi, int m_lo, int m_hi);

    /// Copy u_* → u0_* for all cells (including ghosts).
    void copy_u_to_u0();
};
