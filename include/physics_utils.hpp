#pragma once

#include "fields.hpp"
#include "grid.hpp"

/// Rebuild all 8 conservative variables u_* at cell (l, m) from the physical
/// variables already written at that cell.
///
/// Called by every boundary-condition implementation that sets physical
/// variables (rho, v_*, H_*, e) and needs the conservative u_* arrays to
/// match before the end-of-step update_physical_from_u() pass.
///
/// This is a free inline function rather than a Solver member so that any
/// IBoundaryCondition implementation can call it without depending on Solver.
inline void RebuildUFromPhysical(Fields& f, const Grid& g, int l, int m) noexcept {
    f.u_1[l][m] = f.rho  [l][m] * g.r[l][m];
    f.u_2[l][m] = f.rho  [l][m] * f.v_z  [l][m] * g.r[l][m];
    f.u_3[l][m] = f.rho  [l][m] * f.v_r  [l][m] * g.r[l][m];
    f.u_4[l][m] = f.rho  [l][m] * f.v_phi[l][m] * g.r[l][m];
    f.u_5[l][m] = f.rho  [l][m] * f.e    [l][m] * g.r[l][m];
    f.u_6[l][m] = f.H_phi[l][m];
    f.u_7[l][m] = f.H_z  [l][m] * g.r[l][m];
    f.u_8[l][m] = f.H_r  [l][m] * g.r[l][m];
}
