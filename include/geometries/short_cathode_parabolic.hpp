#pragma once
#include "igeometry.hpp"
namespace YAML { class Node; }

/// Short cathode geometry with a parabolic-smoothed inner-wall transition.
///
/// Outer wall: r_outer(z) = 0.8  (constant)
///
/// Inner wall: piecewise profile with a throat at z ≈ 0.4:
///   z < 0.3           r_inner = 0.2
///   0.3 ≤ z < 0.4     r_inner = 0.2 - 10(z-0.3)²
///   0.4 ≤ z < 0.478   r_inner = 10(z-0.5)²
///   z ≥ 0.478         r_inner = 0.005  (thin inner electrode)
///
/// The params YAML node is currently unused (geometry is fixed), but is
/// accepted for API consistency; future versions may add r_outer, throat_z, etc.
class ShortCathodeParabolicGeometry : public IGeometry {
public:
    explicit ShortCathodeParabolicGeometry(const YAML::Node& /*params*/) {}

    [[nodiscard]] auto RInner    (double z) const -> double override;
    [[nodiscard]] auto ROuter    (double z) const -> double override;
    [[nodiscard]] auto DrInnerDz(double z) const -> double override;
    [[nodiscard]] auto DrOuterDz(double z) const -> double override;

    [[nodiscard]] auto Name() const -> std::string override { return "short_cathode_parabolic"; }
};
