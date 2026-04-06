#include "bcs/solid_wall_bc.hpp"

#include "bc_context.hpp"
#include "physics_utils.hpp"

void SolidWallBC::apply(BCContext& ctx) const {
    Fields&      f = ctx.fields;
    const Grid&  g = ctx.grid;

    constexpr int m = 1;   // fixed index for M_LO face

    #pragma omp parallel for
    for (int l = ctx.local_lo; l <= ctx.local_hi; ++l) {
        // Scalar fields: zero-gradient from the first interior cell above the wall.
        f.rho  [l][m] = f.rho  [l][m + 1];
        f.v_z  [l][m] = f.v_z  [l][m + 1];
        f.v_phi[l][m] = f.v_phi[l][m + 1];
        f.e    [l][m] = f.e    [l][m + 1];
        f.H_phi[l][m] = f.H_phi[l][m + 1];
        f.H_z  [l][m] = f.H_z  [l][m + 1];

        // Wall-tangent condition: radial components follow the wall slope at m+1.
        f.v_r  [l][m] = f.v_z[l][m + 1] * g.r_z[l][m + 1];
        f.H_r  [l][m] = f.H_z[l][m + 1] * g.r_z[l][m + 1];

        rebuild_u_from_physical(f, g, l, m);
    }
}
