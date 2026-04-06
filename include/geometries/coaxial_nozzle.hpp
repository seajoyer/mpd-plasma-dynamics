#pragma once
#include "igeometry.hpp"
namespace YAML { class Node; }

/// Coaxial nozzle-channel geometry (the original hard-coded profile).
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
class CoaxialNozzleGeometry : public IGeometry {
public:
    explicit CoaxialNozzleGeometry(const YAML::Node& /*params*/) {}

    double r_inner    (double z) const override;
    double r_outer    (double z) const override;
    double dr_inner_dz(double z) const override;
    double dr_outer_dz(double z) const override;

    std::string name() const override { return "coaxial_nozzle"; }
};
