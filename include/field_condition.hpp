#pragma once

#include <string>

/// Condition applied to one physical field at a boundary segment.
///
/// Condition types
/// ───────────────
///   Neumann     Zero-gradient by default: f_bc = f_interior + offset
///               where offset defaults to 0 (pure zero-gradient copy).
///               Set FieldCond::value to a non-zero constant to shift the
///               boundary value away from the neighbour, or use an Expression
///               for a fully general condition.
///
///               YAML forms:
///                 neumann                  →  offset = 0  (zero-gradient copy)
///                 { neumann: 0.5 }         →  f_bc = f_nb + 0.5
///
///   Dirichlet   Fix boundary cell to a constant.  FieldCond::value holds the
///               fixed value (defaults to 0 when the bare keyword is used).
///
///               YAML forms:
///                 dirichlet                →  fix to 0
///                 { dirichlet: 1.0 }       →  fix to 1.0
///
///   Expression  Evaluate an arbitrary mathematical formula to compute the
///               boundary-cell value.  The expression string is stored in
///               FieldCond::expr_str and compiled once at construction time
///               by PerFieldBC via the exprtk library.
///
///               Symbols available inside expressions
///               ─────────────────────────────────────
///                Physics constants (read-only, from the [physics] block):
///                  gamma   beta   H_z0   r_0
///
///                Cell spacings (read-only):
///                  dz   dr
///
///                Spatial at the boundary cell:
///                  r   r_z   z
///
///                Spatial at the interior-neighbour cell:
///                  r_nb   r_z_nb
///
///                Math constant:
///                  pi
///
///                Interior-neighbour field values (the cell immediately inside
///                the domain boundary, used for Neumann-like expressions):
///                  rho_nb  v_z_nb  v_r_nb  v_phi_nb  e_nb
///                  H_z_nb  H_r_nb  H_phi_nb
///
///                Previously-evaluated boundary-cell values (set in evaluation
///                order so later fields can reference earlier ones):
///                  rho → v_z → v_r → v_phi → H_z → H_r → H_phi → e
///                  rho  v_z  v_r  v_phi  H_z  H_r  H_phi  e
///
///                Standard math functions (always available):
///                  sin  cos  tan  asin  acos  atan  atan2
///                  exp  log  log2  log10  sqrt  pow  abs  sign
///                  min  max  clamp
///                  (Use exp(1) for Euler's number — 'e' is specific energy.)
///
///               YAML forms:
///                 "v_z * r_z"                  →  bare quoted expression
///                 "r_0 / r"
///                 { expr: "v_z_nb * r_z_nb" }  →  explicit map form
///
///               Migration guide — former hardcoded presets and their
///               expression equivalents:
///
///                 Former keyword         Expression equivalent
///                 ────────────────────────────────────────────
///                 wall_tangent on M_LO → "v_z_nb * r_z_nb"
///                   (M_LO historically used the interior-neighbour slope)
///                 wall_tangent on M_HI → "v_z * r_z"
///                   (M_HI used the wall-cell slope)
///                 hphi_r0_over_r       → "r_0 / r"
///
///   AxisLF      One-sided Lax–Friedrichs update at the axis of symmetry.
///               Applies the half-stencil formula for one of the four
///               conserved variables that receive an LF update (u_1, u_2,
///               u_5, u_7), corresponding to physical fields rho, v_z, e,
///               and H_z respectively.  Valid on the M_LO face only.
///
///               The conserved u_* array is written directly; the physical
///               variable is reconstructed at the end of Solver::advance()
///               by Fields::update_physical_from_u(), so no explicit physical
///               assignment is needed here.
///
///               To reproduce the full axis-of-symmetry condition use AxisLF
///               for rho / v_z / e / H_z and Dirichlet(0) for
///               v_r / v_phi / H_r / H_phi.
enum class FieldCondType {
    Neumann,
    Dirichlet,
    Expression,
    AxisLF,
};

/// Condition configuration for one physical field at one segment.
struct FieldCond {
    FieldCondType type     = FieldCondType::Neumann;
    double        value    = 0.0;   ///< Neumann: additive offset from nb value (0 = zero-gradient).
                                    ///< Dirichlet: fixed boundary value.
    std::string   expr_str;         ///< Expression type only: the formula string.
};
