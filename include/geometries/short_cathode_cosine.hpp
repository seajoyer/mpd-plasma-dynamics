#pragma once
#include "igeometry.hpp"
namespace YAML { class Node; }

/// Short cathode geometry with a cosine-smoothed inner-wall transition.
///
/// Outer wall: r_outer(z) = 0.8  (anode, constant)
///
/// Inner wall: cosine blend from r = 0.2 (cathode) to r = 0.005 (axis):
///
///   z < z_start          r_inner = 0.200
///   z_start ≤ z < z_end  r_inner = 0.200 + (0.005 - 0.200) · ½(1 − cos(π·ξ))
///   z ≥ z_end            r_inner = 0.005
///
/// where  z_center = 0.31,  transition_width = 0.015,
///        z_start  = z_center − transition_width = 0.295,
///        z_end    = z_center + transition_width = 0.325,
///        ξ = (z − z_start) / (z_end − z_start) ∈ [0, 1].
///
/// The derivative dr_inner/dz is C¹ continuous everywhere and zero outside
/// the transition band, making this profile better conditioned than the
/// piecewise-parabolic coaxial_nozzle profile near the throat.
///
/// Recommended boundary-condition split for this geometry (in config.yaml):
///   m_lo:
///     - range: [0, 260]     # z ≤ 0.325 — solid cathode wall
///       type: solid_wall
///     - range: [261, -1]    # z > 0.325 — open axis of symmetry
///       type: axis_symmetry
///
/// (With L_max_global = 800 and dz = 1/800, global l = 260 → z = 0.325.)
class ShortCathodeCosineGeometry : public IGeometry {
public:
    explicit ShortCathodeCosineGeometry(const YAML::Node& /*params*/) {}

    double r_inner    (double z) const override;
    double r_outer    (double z) const override;
    double dr_inner_dz(double z) const override;
    double dr_outer_dz(double z) const override;

    std::string name() const override { return "short_cathode_cosine"; }
};
