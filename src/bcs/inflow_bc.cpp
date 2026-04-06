#include "bcs/inflow_bc.hpp"

#include <cmath>
#include "bc_context.hpp"
#include "physics_utils.hpp"

void InflowBC::apply(BCContext& ctx) const {
    Fields&          f   = ctx.fields;
    const Grid&      g   = ctx.grid;
    const SimConfig& cfg = ctx.cfg;

    const double gamma = cfg.gamma;
    const double beta  = cfg.beta;
    const double H_z0  = cfg.H_z0;
    const double r_0   = g.r_0;

    constexpr int l = 1;   // fixed index for L_LO face

    #pragma omp parallel for
    for (int m = ctx.local_lo; m <= ctx.local_hi; ++m) {
        f.rho  [l][m] = 1.0;
        f.v_phi[l][m] = 0.0;
        f.v_r  [l][m] = 0.0;

        // v_z is taken from the interior so that mass flux is self-consistent.
        f.v_z  [l][m] = f.u_2[2][m] / (f.rho[l][m] * g.r[l][m]);

        f.H_z  [l][m] = H_z0;
        f.H_r  [l][m] = 0.0;
        f.H_phi[l][m] = r_0 / g.r[l][m];

        f.e    [l][m] = beta / (2.0 * (gamma - 1.0))
                        * std::pow(f.rho[l][m], gamma - 1.0);

        rebuild_u_from_physical(f, g, l, m);
    }
}
