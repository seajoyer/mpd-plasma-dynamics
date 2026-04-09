#pragma once

#include <memory>
#include <string>

#include "iinitial_condition.hpp"

namespace YAML { class Node; }

/// Expression-based initial condition — the sole built-in IC type.
///
/// Every primary physical field (rho, v_z, v_r, v_phi, H_z, H_r, H_phi, e)
/// is specified as a mathematical expression string evaluated cell-by-cell at
/// initialisation time.  Derived fields (p, P) are computed automatically from
/// the primaries after all eight have been assigned.
///
/// ═══════════════════════════════════════════════════════════════════
/// YAML structure
/// ═══════════════════════════════════════════════════════════════════
///
///   initial_conditions:
///     type: expression
///     params:
///
///       # ── Optional: user-defined named constants ───────────────
///       vars:
///         <name>: <double>
///         ...
///
///       # ── Field expressions (all optional; see defaults below) ──
///       rho:   <expr>
///       v_z:   <expr>
///       v_r:   <expr>
///       v_phi: <expr>
///       H_z:   <expr>
///       H_r:   <expr>
///       H_phi: <expr>
///       e:     <expr>
///
/// ═══════════════════════════════════════════════════════════════════
/// Symbols available in every expression
/// ═══════════════════════════════════════════════════════════════════
///
///   Physics constants  (read-only; taken from the [physics] block)
///     gamma      adiabatic index
///     beta       plasma beta  (p / (B²/2μ₀))
///     H_z0       reference axial magnetic field
///     r_0        mid-channel radius at z = 0:  (r_inner(0) + r_outer(0)) / 2
///
///   Spatial variables  (updated per cell, read-only in expressions)
///     z          axial coordinate   [l_global × dz]
///     r          radial coordinate  [grid.r[l][m]]
///     r_z        dr/dz skew metric  [grid.r_z[l][m]]
///
///   Mathematical constant
///     pi         3.141592653589793
///
///   Standard math functions (always available)
///     sin  cos  tan  asin  acos  atan  atan2
///     exp  log  log2 log10 sqrt  pow   abs   sign
///     min  max  clamp
///     Note: use exp(1) for Euler's number — 'e' is the specific-energy field.
///
///   User-defined constants  (declared under params.vars; read-only)
///     <name>     the value specified in the vars block
///
///   Previously-assigned field values  (updated in evaluation order)
///     rho   v_z   v_r   v_phi   H_z   H_r   H_phi   e
///     A field expression may reference any field earlier in the sequence.
///     Referencing a later field yields 0.0 (its uninitialised default).
///
/// ═══════════════════════════════════════════════════════════════════
/// Evaluation order
/// ═══════════════════════════════════════════════════════════════════
///
///   rho → v_z → v_r → v_phi → H_z → H_r → H_phi → e
///
/// ═══════════════════════════════════════════════════════════════════
/// Default expressions  (used when a field key is absent)
/// ═══════════════════════════════════════════════════════════════════
///
///   rho:   1.0
///   v_z:   0.0
///   v_r:   0.0
///   v_phi: 0.0
///   H_z:   H_z0
///   H_r:   "H_z * r_z"
///   H_phi: "(1 - 0.9 * z) * r_0 / r"
///   e:     "beta / (2 * (gamma - 1))"
///
/// ═══════════════════════════════════════════════════════════════════
/// YAML quoting rules
/// ═══════════════════════════════════════════════════════════════════
///
///   Quote any expression that contains YAML-special characters: * { } : [ ]
///   Plain numbers (1.0, 0.25) and bare identifiers (H_z0) need no quotes.
///
/// ═══════════════════════════════════════════════════════════════════
/// User-defined constants  (params.vars)
/// ═══════════════════════════════════════════════════════════════════
///
///   Constants declared under params.vars are registered before any expression
///   is compiled, so they can appear in every field expression.  Names must
///   not shadow built-in symbols (gamma, beta, H_z0, r_0, z, r, r_z, pi,
///   rho, v_z, v_r, v_phi, H_z, H_r, H_phi, e).  A runtime error is thrown
///   at startup if a conflict is detected.
///
/// ═══════════════════════════════════════════════════════════════════
/// Examples
/// ═══════════════════════════════════════════════════════════════════
///
///   # Minimal: only specify what differs from the defaults.
///   initial_conditions:
///     type: expression
///     params:
///       v_z: 0.1
///
///   # Free-vortex azimuthal field with a user-controlled amplitude.
///   initial_conditions:
///     type: expression
///     params:
///       vars:
///         amp: 0.8
///       H_phi: "amp * r_0 / r"
///
///   # Radially stratified density with a Gaussian jet velocity profile.
///   initial_conditions:
///     type: expression
///     params:
///       vars:
///         rho_wall: 2.0
///         jet_v:    0.5
///         sigma:    0.03
///       rho:  "1.0 + (rho_wall - 1.0) * (r - r_0) / r_0"
///       v_z:  "jet_v * exp(-((r - r_0)^2) / (2 * sigma^2))"
///       H_r:  "H_z * r_z"
///       H_phi: "(1 - 0.9 * z) * r_0 / r"
class ExpressionIC : public IInitialCondition {
public:
    explicit ExpressionIC(const YAML::Node& params);

    // Defined in .cpp so the compiler sees the complete Impl type for deletion.
    ~ExpressionIC() override;

    void Apply(Fields& fields, const Grid& grid,
               const SimConfig& cfg, int l_start) const override;

    [[nodiscard]] auto Name() const -> std::string override { return "expression"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
