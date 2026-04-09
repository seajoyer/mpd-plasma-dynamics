#pragma once

#include "field_ic.hpp"
#include "iinitial_condition.hpp"

namespace YAML { class Node; }

/// Built-in initial condition that applies an independent FieldIC to each
/// of the eight primary physical variables (rho, v_z, v_r, v_phi, e, H_z,
/// H_r, H_phi).  Derived scalar fields p and P are computed consistently
/// from the primaries after all eight have been set.
///
/// All fields are applied cell-by-cell over the interior domain, so spatial
/// profiles (FreeVortex, LinearFreeVortex, WallTangent) are naturally
/// supported alongside uniform constants.
///
/// Per-field YAML specification (all keys are optional)
/// ─────────────────────────────────────────────────────
///   rho / v_z / v_r / v_phi / e / H_z / H_r / H_phi
///
///   Each key accepts one of the following forms:
///
///     from_physics              Scalar string — derive from cfg physics params.
///                               H_z  → cfg.H_z0
///                               e    → cfg.beta / (2·(cfg.gamma − 1))
///
///     wall_tangent              Scalar string — H_r = H_z · r_z.
///                               Valid only for H_r.
///
///     linear_free_vortex        Scalar string — H_phi = (1 − 0.9·z) · r_0/r.
///                               Uses default amplitude = 1.0, factor = 0.9.
///
///     { uniform: <v> }          Map — constant value v everywhere.
///
///     { free_vortex: <a> }      Map — H_phi = a · r_0/r.
///     { free_vortex: { amplitude: <a> } }
///
///     { linear_free_vortex: { amplitude: <a>, factor: <f> } }
///                               Map — H_phi = a · (1 − f·z) · r_0/r.
///
/// Default per field when the key is absent
/// ─────────────────────────────────────────
///   rho   → uniform 1.0
///   v_z   → uniform 0.0
///   v_r   → uniform 0.0
///   v_phi → uniform 0.0
///   e     → from_physics    [cfg.beta / (2·(cfg.gamma − 1))]
///   H_z   → from_physics    [cfg.H_z0]
///   H_r   → wall_tangent    [H_z · r_z]
///   H_phi → linear_free_vortex  [amplitude=1, factor=0.9]
class UniformMhdIC : public IInitialCondition {
public:
    explicit UniformMhdIC(const YAML::Node& params);

    void Apply(Fields& fields, const Grid& grid,
               const SimConfig& cfg, int l_start) const override;

    [[nodiscard]] auto Name() const -> std::string override { return "uniform_mhd"; }

private:
    FieldIC rho_;
    FieldIC v_z_;
    FieldIC v_r_;
    FieldIC v_phi_;
    FieldIC e_;
    FieldIC H_z_;
    FieldIC H_r_;
    FieldIC H_phi_;
};
