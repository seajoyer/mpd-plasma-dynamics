#pragma once

#include <string>

/// Abstract interface for the channel geometry.
///
/// Provides the inner and outer radial wall profiles as functions of the
/// axial coordinate z, together with their z-derivatives (needed by Grid to
/// compute the mesh-skew metric r_z = dr/dz).
///
/// Implementations are registered by name in GeometryRegistry and selected at
/// runtime via the `geometry.type` field in config.yaml.
///
/// Lifetime requirement: the geometry object must outlive every Grid that
/// holds a reference to it (both are typically owned by main()).
class IGeometry {
public:
    virtual ~IGeometry() = default;

    /// Inner-wall radius at axial position z.
    [[nodiscard]] virtual auto RInner   (double z) const -> double = 0;

    /// Outer-wall radius at axial position z.
    [[nodiscard]] virtual auto ROuter   (double z) const -> double = 0;

    /// d(r_inner)/dz — used for the mesh-skew metric.
    [[nodiscard]] virtual auto DrInnerDz(double z) const -> double = 0;

    /// d(r_outer)/dz — used for the mesh-skew metric.
    [[nodiscard]] virtual auto DrOuterDz(double z) const -> double = 0;

    /// Human-readable name, used in log messages.
    [[nodiscard]] virtual auto Name() const -> std::string = 0;
};
