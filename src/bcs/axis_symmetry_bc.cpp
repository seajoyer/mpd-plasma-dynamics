#include "bcs/axis_symmetry_bc.hpp"

#include "bc_context.hpp"

void AxisSymmetryBC::apply(BCContext& ctx) const {
    Fields&           f   = ctx.fields;
    const Grid&       g   = ctx.grid;
    const SimConfig&  cfg = ctx.cfg;
    const double      dt  = ctx.dt;
    const double      dz  = cfg.dz;

    auto** u0_1 = f.u0_1.raw(); auto** u0_2 = f.u0_2.raw();
    auto** u0_5 = f.u0_5.raw(); auto** u0_7 = f.u0_7.raw();
    auto** v_z  = f.v_z.raw();
    auto** v_r  = f.v_r.raw();
    auto** p    = f.p.raw();
    auto** P    = f.P.raw();
    auto** H_z  = f.H_z.raw();
    auto** H_r  = f.H_r.raw();
    auto** r    = g.r.raw();
    const double* dr = g.dr.data();

    constexpr int m = 1;   // fixed index for M_LO face

    // Axis-of-symmetry conditions:
    //   v_r = 0,  v_phi = 0,  H_phi = 0,  H_r = 0
    //
    // The one-sided stencil in m treats the virtual mirror cell at m=0 as
    // having the same values as m=1 (symmetry plane), which cancels the
    // m-direction finite-difference terms for the symmetric quantities and
    // halves the contribution from the radial gradient for the rest.
    //
    // The update mirrors apply_bc_lower_outer() from the original solver,
    // using the local-to-r division form to handle the cylindrical metric.

    #pragma omp parallel for
    for (int l = ctx.local_lo; l <= ctx.local_hi; ++l) {
        const double dr_l = dr[l];

        // u1: ρ·r — mass conservation
        f.u_1[l][m] =
            (0.25 * (u0_1[l+1][m]/r[l+1][m] + u0_1[l-1][m]/r[l-1][m]
                   + u0_1[l][m+1]/r[l][m+1]  + u0_1[l][m]/r[l][m])
             + dt * (-(u0_1[l+1][m]/r[l+1][m]*v_z[l+1][m]
                      -u0_1[l-1][m]/r[l-1][m]*v_z[l-1][m]) / (2*dz)
                     -(u0_1[l][m+1]/r[l][m+1]*v_r[l][m+1]
                      -u0_1[l][m]  /r[l][m]  *v_r[l][m+1]) / dr_l))
            * r[l][m];

        // u2: ρ·v_z·r — axial momentum
        f.u_2[l][m] =
            (0.25 * (u0_2[l+1][m]/r[l+1][m] + u0_2[l-1][m]/r[l-1][m]
                   + u0_2[l][m+1]/r[l][m+1]  + u0_2[l][m]/r[l][m])
             + dt * (( (H_z[l+1][m]*H_z[l+1][m] - P[l+1][m])
                      -(H_z[l-1][m]*H_z[l-1][m] - P[l-1][m])) / (2*dz)
                    +( H_z[l][m+1]*H_r[l][m+1] - H_z[l][m]*H_r[l][m]) / dr_l
                     -(u0_2[l+1][m]/r[l+1][m]*v_z[l+1][m]
                      -u0_2[l-1][m]/r[l-1][m]*v_z[l-1][m]) / (2*dz)
                     -(u0_2[l][m+1]/r[l][m+1]*v_r[l][m+1]
                      -u0_2[l][m]  /r[l][m]  *v_r[l][m])    / dr_l))
            * r[l][m];

        // Symmetry conditions: zero radial / azimuthal momentum and field.
        f.u_3[l][m] = 0.0;
        f.u_4[l][m] = 0.0;

        // u5: ρ·e·r — internal energy
        f.u_5[l][m] =
            (0.25 * (u0_5[l+1][m]/r[l+1][m] + u0_5[l-1][m]/r[l-1][m]
                   + u0_5[l][m+1]/r[l][m+1]  + u0_5[l][m]/r[l][m])
             + dt * (-p[l][m] * ((v_z[l+1][m] - v_z[l-1][m]) / (2*dz)
                                +(v_r[l][m+1]  - v_r[l][m])    / dr_l)
                     -(u0_5[l+1][m]/r[l+1][m]*v_z[l+1][m]
                      -u0_5[l-1][m]/r[l-1][m]*v_z[l-1][m]) / (2*dz)
                     -(u0_5[l][m+1]/r[l][m+1]*v_r[l][m+1]
                      -u0_5[l][m]  /r[l][m]  *v_r[l][m])    / dr_l))
            * r[l][m];

        // Symmetry condition: zero toroidal field.
        f.u_6[l][m] = 0.0;

        // u7: H_z·r — axial magnetic flux
        f.u_7[l][m] =
            (0.25 * (u0_7[l+1][m]/r[l+1][m] + u0_7[l-1][m]/r[l-1][m]
                   + u0_7[l][m+1]/r[l][m+1]  + u0_7[l][m]/r[l][m])
             + dt * ((H_r[l][m+1]*v_z[l][m+1] - H_r[l][m]*v_z[l][m]) / dr_l
                     -(u0_7[l][m+1]/r[l][m+1]*v_r[l][m+1]
                      -u0_7[l][m]  /r[l][m]  *v_r[l][m])               / dr_l))
            * r[l][m];

        // Symmetry condition: zero radial magnetic flux.
        f.u_8[l][m] = 0.0;
    }
    // Physical variables will be reconstructed from u_* by
    // Fields::update_physical_from_u() at the end of Solver::advance().
}
