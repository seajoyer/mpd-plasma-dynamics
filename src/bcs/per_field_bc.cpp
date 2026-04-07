#include "bcs/per_field_bc.hpp"

#include <cmath>
#include <stdexcept>

#include "bc_context.hpp"
#include "physics_utils.hpp"

// ============================================================
// Constructor
// ============================================================

PerFieldBC::PerFieldBC(FaceBC::Face face, const BCSegmentConfig& seg)
    : face_(face),
      rho_  (seg.rho),
      v_z_  (seg.v_z),
      v_r_  (seg.v_r),
      v_phi_(seg.v_phi),
      e_    (seg.e),
      H_z_  (seg.H_z),
      H_r_  (seg.H_r),
      H_phi_(seg.H_phi),
      has_axis_lf_(
          seg.rho.type  == FieldCondType::AxisLF ||
          seg.v_z.type  == FieldCondType::AxisLF ||
          seg.e.type    == FieldCondType::AxisLF ||
          seg.H_z.type  == FieldCondType::AxisLF)
{
    // Validate AxisLF is only used on the M_LO face.
    if (has_axis_lf_ && face_ != FaceBC::Face::M_LO)
        throw std::runtime_error(
            "PerFieldBC: AxisLF condition is only valid on the M_LO face "
            "(inner boundary / axis of symmetry)");

    // Validate WallTangent is only used on M faces.
    const bool wall_tangent_used =
        seg.v_r.type == FieldCondType::WallTangent ||
        seg.H_r.type == FieldCondType::WallTangent;
    if (wall_tangent_used &&
        face_ != FaceBC::Face::M_LO && face_ != FaceBC::Face::M_HI)
        throw std::runtime_error(
            "PerFieldBC: WallTangent condition is only valid on M_LO / M_HI faces");

    // Validate HPhi_r0_over_r is only applied to H_phi.
    const bool hphi_r0_on_other =
        seg.rho.type   == FieldCondType::HPhi_r0_over_r ||
        seg.v_z.type   == FieldCondType::HPhi_r0_over_r ||
        seg.v_r.type   == FieldCondType::HPhi_r0_over_r ||
        seg.v_phi.type == FieldCondType::HPhi_r0_over_r ||
        seg.e.type     == FieldCondType::HPhi_r0_over_r ||
        seg.H_z.type   == FieldCondType::HPhi_r0_over_r ||
        seg.H_r.type   == FieldCondType::HPhi_r0_over_r;
    if (hphi_r0_on_other)
        throw std::runtime_error(
            "PerFieldBC: HPhi_r0_over_r condition is only valid for the H_phi field");
}

// ============================================================
// apply
// ============================================================

void PerFieldBC::apply(BCContext& ctx) const {
    Fields&           f   = ctx.fields;
    const Grid&       g   = ctx.grid;
    const SimConfig&  cfg = ctx.cfg;
    const MPIManager& mpi = ctx.mpi;
    const double      dt  = ctx.dt;
    const double      dz  = cfg.dz;

    // ---- Determine fixed index and interior-neighbour index ----------------
    // For L faces: fixed axis is l, free axis is m.
    // For M faces: fixed axis is m, free axis is l.
    const bool is_l_face =
        (face_ == FaceBC::Face::L_LO || face_ == FaceBC::Face::L_HI);

    int l_fix{0}, m_fix{0};   // boundary cell index (fixed axis)
    int l_nb{0},  m_nb{0};    // interior neighbour index (for Neumann)

    switch (face_) {
        case FaceBC::Face::L_LO:
            l_fix = 1;             l_nb = 2;
            break;
        case FaceBC::Face::L_HI:
            l_fix = mpi.local_L;   l_nb = mpi.local_L - 1;
            break;
        case FaceBC::Face::M_LO:
            m_fix = 1;             m_nb = 2;
            break;
        case FaceBC::Face::M_HI:
            m_fix = mpi.local_M;   m_nb = mpi.local_M - 1;
            break;
    }

    // ---- Helper: resolve one scalar condition at a single cell -------------
    // Returns the value to write into the boundary cell.
    // WallTangent, AxisLF, and HPhi_r0_over_r are handled separately (they
    // need more context), so this helper returns the current cell value
    // unchanged for those types.
    auto resolve = [](const FieldCond& c, double cell, double nb) -> double {
        switch (c.type) {
            case FieldCondType::Neumann:   return nb;
            case FieldCondType::Dirichlet: return c.value;
            default:                       return cell;   // handled externally
        }
    };

    // ========================================================
    // L faces — iterate over m, fixed l
    // ========================================================
    if (is_l_face) {
        const int l  = l_fix;
        const int ln = l_nb;

        #pragma omp parallel for
        for (int m = ctx.local_lo; m <= ctx.local_hi; ++m) {
            // --- Step 1: apply physical-variable conditions ---
            f.rho  [l][m] = resolve(rho_,   f.rho  [l][m], f.rho  [ln][m]);
            f.v_z  [l][m] = resolve(v_z_,   f.v_z  [l][m], f.v_z  [ln][m]);
            f.v_r  [l][m] = resolve(v_r_,   f.v_r  [l][m], f.v_r  [ln][m]);
            f.v_phi[l][m] = resolve(v_phi_, f.v_phi[l][m], f.v_phi[ln][m]);
            f.e    [l][m] = resolve(e_,     f.e    [l][m], f.e    [ln][m]);
            f.H_z  [l][m] = resolve(H_z_,   f.H_z  [l][m], f.H_z  [ln][m]);
            f.H_r  [l][m] = resolve(H_r_,   f.H_r  [l][m], f.H_r  [ln][m]);

            // H_phi: free-vortex profile H_phi = r_0 / r  (enforces H_phi·r = const)
            if (H_phi_.type == FieldCondType::HPhi_r0_over_r)
                f.H_phi[l][m] = g.r_0 / g.r[l][m];
            else
                f.H_phi[l][m] = resolve(H_phi_, f.H_phi[l][m], f.H_phi[ln][m]);

            // --- Step 2: rebuild conservative vars from physical ------------
            rebuild_u_from_physical(f, g, l, m);
        }
        return;
    }

    // ========================================================
    // M faces — iterate over l, fixed m
    // ========================================================

    const int m      = m_fix;
    const int mn     = m_nb;
    const bool is_lo = (face_ == FaceBC::Face::M_LO);

    // For WallTangent the slope is evaluated:
    //   M_LO: at the interior-side neighbour cell (mn = 2), matching the
    //         original solid_wall_bc which used r_z[l][m+1].
    //   M_HI: at the wall cell itself (m = local_M), matching outer_wall_bc.
    const int m_slope = is_lo ? mn : m;

    #pragma omp parallel for
    for (int l = ctx.local_lo; l <= ctx.local_hi; ++l) {

        // --- Step 1: apply physical-variable conditions (scalar fields) -----
        f.rho  [l][m] = resolve(rho_,   f.rho  [l][m], f.rho  [l][mn]);
        f.v_phi[l][m] = resolve(v_phi_, f.v_phi[l][m], f.v_phi[l][mn]);
        f.e    [l][m] = resolve(e_,     f.e    [l][m], f.e    [l][mn]);

        // H_phi: free-vortex profile or standard conditions
        if (H_phi_.type == FieldCondType::HPhi_r0_over_r)
            f.H_phi[l][m] = g.r_0 / g.r[l][m];
        else
            f.H_phi[l][m] = resolve(H_phi_, f.H_phi[l][m], f.H_phi[l][mn]);

        // v_z and H_z must be set before wall-tangent v_r / H_r.
        f.v_z[l][m] = resolve(v_z_, f.v_z[l][m], f.v_z[l][mn]);
        f.H_z[l][m] = resolve(H_z_, f.H_z[l][m], f.H_z[l][mn]);

        // Radial velocity ----
        if (v_r_.type == FieldCondType::WallTangent)
            f.v_r[l][m] = f.v_z[l][m_slope] * g.r_z[l][m_slope];
        else
            f.v_r[l][m] = resolve(v_r_, f.v_r[l][m], f.v_r[l][mn]);

        // Radial magnetic field ----
        if (H_r_.type == FieldCondType::WallTangent)
            f.H_r[l][m] = f.H_z[l][m_slope] * g.r_z[l][m_slope];
        else
            f.H_r[l][m] = resolve(H_r_, f.H_r[l][m], f.H_r[l][mn]);

        // --- Step 2: rebuild conservative vars from physical ----------------
        // This sets all u_* from the physical values written above.
        // For AxisLF fields, u_* will be overwritten in step 3 — the values
        // set here use stale physical vars for those fields, but they are
        // immediately superseded and never observed.
        rebuild_u_from_physical(f, g, l, m);

        // --- Step 3: AxisLF stencil (overwrites u_* for LF-updated fields) --
        if (has_axis_lf_) {
            if (rho_.type == FieldCondType::AxisLF)
                f.u_1[l][m] = axis_lf_u1(f, g, l, m, dt, dz);
            if (v_z_.type == FieldCondType::AxisLF)
                f.u_2[l][m] = axis_lf_u2(f, g, l, m, dt, dz);
            if (e_.type == FieldCondType::AxisLF)
                f.u_5[l][m] = axis_lf_u5(f, g, l, m, dt, dz);
            if (H_z_.type == FieldCondType::AxisLF)
                f.u_7[l][m] = axis_lf_u7(f, g, l, m, dt, dz);
            // Physical vars for AxisLF fields are reconstructed at the end of
            // Solver::advance() by Fields::update_physical_from_u().
        }
    }
}

// ============================================================
// AxisLF stencil helpers
//
// Each function mirrors the corresponding update in the original
// axis_symmetry_bc.cpp.  The formulas implement a one-sided Lax–Friedrichs
// step at m = 1 where the virtual "inner" neighbour at m = 0 is treated as
// the mirror image of m = 1 (axis-symmetry condition), which cancels the
// m-direction flux term for that side.
// ============================================================

double PerFieldBC::axis_lf_u1(const Fields& f, const Grid& g,
                               int l, int m, double dt, double dz) {
    auto** u0 = f.u0_1.raw();
    auto** vz = f.v_z.raw();
    auto** vr = f.v_r.raw();
    auto** r  = g.r.raw();
    const double dr_l = g.dr[l];

    return (0.25 * (u0[l+1][m]/r[l+1][m] + u0[l-1][m]/r[l-1][m]
                  + u0[l][m+1]/r[l][m+1]  + u0[l][m]  /r[l][m])
            + dt * (-(u0[l+1][m]/r[l+1][m]*vz[l+1][m]
                     -u0[l-1][m]/r[l-1][m]*vz[l-1][m]) / (2.0*dz)
                    -(u0[l][m+1]/r[l][m+1]*vr[l][m+1]
                     -u0[l][m]  /r[l][m]  *vr[l][m+1]) / dr_l))
           * r[l][m];
}

double PerFieldBC::axis_lf_u2(const Fields& f, const Grid& g,
                               int l, int m, double dt, double dz) {
    auto** u0 = f.u0_2.raw();
    auto** vz = f.v_z.raw();
    auto** vr = f.v_r.raw();
    auto** Hz = f.H_z.raw();
    auto** Hr = f.H_r.raw();
    auto** P  = f.P.raw();
    auto** r  = g.r.raw();
    const double dr_l = g.dr[l];

    return (0.25 * (u0[l+1][m]/r[l+1][m] + u0[l-1][m]/r[l-1][m]
                  + u0[l][m+1]/r[l][m+1]  + u0[l][m]  /r[l][m])
            + dt * (((Hz[l+1][m]*Hz[l+1][m] - P[l+1][m])
                    -(Hz[l-1][m]*Hz[l-1][m] - P[l-1][m])) / (2.0*dz)
                   +( Hz[l][m+1]*Hr[l][m+1] - Hz[l][m]*Hr[l][m]) / dr_l
                    -(u0[l+1][m]/r[l+1][m]*vz[l+1][m]
                     -u0[l-1][m]/r[l-1][m]*vz[l-1][m]) / (2.0*dz)
                    -(u0[l][m+1]/r[l][m+1]*vr[l][m+1]
                     -u0[l][m]  /r[l][m]  *vr[l][m])    / dr_l))
           * r[l][m];
}

double PerFieldBC::axis_lf_u5(const Fields& f, const Grid& g,
                               int l, int m, double dt, double dz) {
    auto** u0 = f.u0_5.raw();
    auto** vz = f.v_z.raw();
    auto** vr = f.v_r.raw();
    auto** p  = f.p.raw();
    auto** r  = g.r.raw();
    const double dr_l = g.dr[l];

    return (0.25 * (u0[l+1][m]/r[l+1][m] + u0[l-1][m]/r[l-1][m]
                  + u0[l][m+1]/r[l][m+1]  + u0[l][m]  /r[l][m])
            + dt * (-p[l][m] * ((vz[l+1][m] - vz[l-1][m]) / (2.0*dz)
                               +(vr[l][m+1]  - vr[l][m])    / dr_l)
                    -(u0[l+1][m]/r[l+1][m]*vz[l+1][m]
                     -u0[l-1][m]/r[l-1][m]*vz[l-1][m]) / (2.0*dz)
                    -(u0[l][m+1]/r[l][m+1]*vr[l][m+1]
                     -u0[l][m]  /r[l][m]  *vr[l][m])    / dr_l))
           * r[l][m];
}

double PerFieldBC::axis_lf_u7(const Fields& f, const Grid& g,
                               int l, int m, double dt, double dz) {
    auto** u0 = f.u0_7.raw();
    auto** vz = f.v_z.raw();
    auto** vr = f.v_r.raw();
    auto** Hz = f.H_z.raw();
    auto** Hr = f.H_r.raw();
    auto** r  = g.r.raw();
    const double dr_l = g.dr[l];

    return (0.25 * (u0[l+1][m]/r[l+1][m] + u0[l-1][m]/r[l-1][m]
                  + u0[l][m+1]/r[l][m+1]  + u0[l][m]  /r[l][m])
            + dt * ((Hr[l][m+1]*vz[l][m+1] - Hr[l][m]*vz[l][m]) / dr_l
                    -(u0[l][m+1]/r[l][m+1]*vr[l][m+1]
                     -u0[l][m]  /r[l][m]  *vr[l][m])               / dr_l))
           * r[l][m];
}
