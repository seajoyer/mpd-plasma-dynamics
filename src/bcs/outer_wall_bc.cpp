#include "bcs/outer_wall_bc.hpp"

#include "bc_context.hpp"
#include "physics_utils.hpp"

void OuterWallBC::apply(BCContext& ctx) const {
    Fields&          f   = ctx.fields;
    const Grid&      g   = ctx.grid;
    const MPIManager& mpi = ctx.mpi;

    const int m = mpi.local_M;   // fixed index for M_HI face

    #pragma omp parallel for
    for (int l = ctx.local_lo; l <= ctx.local_hi; ++l) {
        // Scalar fields: zero-gradient from interior neighbour.
        f.rho  [l][m] = f.rho  [l][m - 1];
        f.v_z  [l][m] = f.v_z  [l][m - 1];
        f.v_phi[l][m] = f.v_phi[l][m - 1];
        f.e    [l][m] = f.e    [l][m - 1];
        f.H_phi[l][m] = f.H_phi[l][m - 1];
        f.H_z  [l][m] = f.H_z  [l][m - 1];

        // Wall-tangent velocity and magnetic field: derive from slope metric.
        f.v_r  [l][m] = f.v_z[l][m] * g.r_z[l][m];
        f.H_r  [l][m] = f.H_z[l][m] * g.r_z[l][m];

        rebuild_u_from_physical(f, g, l, m);
    }
}
