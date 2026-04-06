#include "bcs/outflow_bc.hpp"

#include "bc_context.hpp"

void OutflowBC::apply(BCContext& ctx) const {
    Fields&          f   = ctx.fields;
    const MPIManager& mpi = ctx.mpi;

    const int l = mpi.local_L;   // fixed index for L_HI face

    #pragma omp parallel for
    for (int m = ctx.local_lo; m <= ctx.local_hi; ++m) {
        f.u_1[l][m] = f.u_1[l - 1][m];
        f.u_2[l][m] = f.u_2[l - 1][m];
        f.u_3[l][m] = f.u_3[l - 1][m];
        f.u_4[l][m] = f.u_4[l - 1][m];
        f.u_5[l][m] = f.u_5[l - 1][m];
        f.u_6[l][m] = f.u_6[l - 1][m];
        f.u_7[l][m] = f.u_7[l - 1][m];
        f.u_8[l][m] = f.u_8[l - 1][m];
    }
    // Physical variables will be reconstructed from u_* by
    // the end-of-step Fields::update_physical_from_u() call in Solver::advance().
}
