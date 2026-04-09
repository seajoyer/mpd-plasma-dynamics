#pragma once

#include <memory>
#include <string>

#include "iinitial_condition.hpp"

namespace YAML { class Node; }

/// Expression-based initial condition.
///
/// Every primary physical field (rho, v_z, v_r, v_phi, H_z, H_r, H_phi, e)
/// is specified as a mathematical expression string that is evaluated
/// cell-by-cell at initialisation time.
///
/// Available symbols
/// ─────────────────
///   Physics constants (read-only)
///     gamma     adiabatic index
///     beta      plasma beta
///     H_z0      reference axial magnetic field
///     r_0       mid-channel radius at z = 0  [ (r_inner(0)+r_outer(0))/2 ]
///
///   Spatial variables (updated per cell)
///     z         axial coordinate of the current cell  [l_global * dz]
///     r         radial coordinate of the current cell [grid.r[l][m]]
///     r_z       dr/dz mesh-skew metric               [grid.r_z[l][m]]
///
///   Mathematical constants
///     pi        3.14159...
///
///   Standard math functions
///     sin, cos, tan, asin, acos, atan, atan2
///     exp, log, log2, log10, sqrt, pow, abs, sign
///     min, max, clamp
///
///   Previously assigned fields (updated in evaluation order)
///     rho, v_z, v_r, v_phi, H_z, H_r, H_phi, e
///
/// Evaluation order
/// ─────────────────
///   rho → v_z → v_r → v_phi → H_z → H_r → H_phi → e
///
///   A field expression may reference any field that appears earlier in
///   this sequence.  Referencing a later field (e.g. using e inside rho's
///   expression) yields the default-initialised value (0.0) rather than a
///   compile-time error; avoid doing so.
///
/// YAML specification
/// ───────────────────
///   Each field accepts a YAML scalar that is either:
///     - A plain number:            rho: 1.0
///     - An expression string:      H_r: "H_z * r_z"
///
///   Quote strings that contain YAML-special characters (*, {, :, etc.).
///   Plain numbers and simple identifiers do not require quoting.
///
///   All eight fields are optional.  Defaults reproduce the standard
///   uniform_mhd physics-derived values:
///
///     rho:   1.0
///     v_z:   0.0
///     v_r:   0.0
///     v_phi: 0.0
///     H_z:   H_z0
///     H_r:   H_z * r_z
///     H_phi: (1 - 0.9 * z) * r_0 / r
///     e:     beta / (2 * (gamma - 1))
///
/// Example
/// ───────
///   initial_conditions:
///     type: expression
///     params:
///       rho:   1.0
///       v_z:   0.1
///       v_r:   0.1
///       v_phi: 0.0
///       H_z:   H_z0
///       H_r:   "H_z * r_z"
///       H_phi: "(1 - 0.9 * z) * r_0 / r"
///       e:     "beta / (2 * (gamma - 1))"
class ExpressionIC : public IInitialCondition {
public:
    explicit ExpressionIC(const YAML::Node& params);

    // Defined in .cpp so the compiler sees the complete Impl type.
    ~ExpressionIC() override;

    void Apply(Fields& fields, const Grid& grid,
               const SimConfig& cfg, int l_start) const override;

    [[nodiscard]] auto Name() const -> std::string override { return "expression"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
