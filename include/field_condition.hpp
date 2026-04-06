#pragma once

/// Condition applied to one physical field at a boundary segment.
///
/// Condition types
/// ───────────────
///   Neumann      Zero-gradient: copy value from the first interior cell
///                (the cell immediately inside the domain boundary).
///
///   Dirichlet    Fixed constant value.  Set FieldCond::value to the
///                desired quantity.
///
///   WallTangent  Derive a radial component from the wall-tangent condition:
///                  v_r = v_z · (dr/dz)    or    H_r = H_z · (dr/dz)
///                Valid only for v_r and H_r on M-direction faces (M_LO / M_HI).
///                The slope metric r_z is evaluated at the interior-side cell
///                for M_LO, and at the wall cell for M_HI, matching the
///                behaviour of the original solid_wall / outer_wall presets.
///
///   AxisLF       One-sided Lax–Friedrichs update at the axis of symmetry.
///                Applies the half-stencil formula for one of the four
///                conserved variables that receive an LF update (u_1, u_2,
///                u_5, u_7), corresponding to physical fields rho, v_z, e,
///                and H_z respectively.  Valid on M_LO face only.
///
///                The conserved u_* array for the field is written directly;
///                the physical variable is reconstructed later by the
///                end-of-step Fields::update_physical_from_u() call in
///                Solver::advance(), so no explicit physical assignment is
///                needed here.
///
///                To reproduce the full axis-of-symmetry condition, use
///                AxisLF for rho / v_z / e / H_z and Dirichlet(0) for
///                v_r / v_phi / H_r / H_phi.
enum class FieldCondType {
    Neumann,
    Dirichlet,
    WallTangent,
    AxisLF,
};

/// Condition configuration for one physical field at one segment.
struct FieldCond {
    FieldCondType type  = FieldCondType::Neumann;
    double        value = 0.0;   ///< Used only for Dirichlet.
};
