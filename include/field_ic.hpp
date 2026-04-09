#pragma once

/// Specification type for one physical field's initial condition.
///
/// Used by UniformMhdIC to configure each of the eight primary physical
/// variables independently via YAML.
///
/// Condition types
/// ───────────────
///   Uniform              Constant value everywhere.
///                        YAML:  { uniform: <double> }
///
///   FromPhysics          Derive the value from SimConfig physics parameters:
///                          H_z  →  cfg.H_z0
///                          e    →  cfg.beta / (2 · (cfg.gamma − 1))
///                        The meaning for other fields is undefined; the
///                        constructor of UniformMhdIC validates this.
///                        YAML:  from_physics
///
///   WallTangent          Radial magnetic component from the wall-slope metric:
///                          H_r = H_z · r_z
///                        Valid only for H_r; H_z must be assigned first.
///                        YAML:  wall_tangent
///
///   FreeVortex           Azimuthal free-vortex profile (constant circulation):
///                          H_phi = amplitude · r_0 / r
///                        Valid only for H_phi.
///                        YAML:  { free_vortex: <amplitude> }
///                               { free_vortex: { amplitude: <double> } }
///
///   LinearFreeVortex     Axially-tapered free vortex:
///                          H_phi = amplitude · (1 − factor · z) · r_0 / r
///                        Valid only for H_phi.
///                        YAML:  linear_free_vortex
///                               { linear_free_vortex: { amplitude: <a>, factor: <f> } }
enum class FieldICType {
    Uniform,
    FromPhysics,
    WallTangent,
    FreeVortex,
    LinearFreeVortex,
};

/// Initial-condition configuration for one physical field.
struct FieldIC {
    FieldICType type      = FieldICType::Uniform;
    double      value     = 0.0;   ///< Uniform value.
    double      amplitude = 1.0;   ///< Scaling for FreeVortex / LinearFreeVortex.
    double      factor    = 0.9;   ///< Axial taper coefficient for LinearFreeVortex.
};
