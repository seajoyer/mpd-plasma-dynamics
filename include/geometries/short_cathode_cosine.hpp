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
/// piecewise-parabolic short_cathode_parabolic profile near the throat.
class ShortCathodeCosineGeometry : public IGeometry {
public:
    explicit ShortCathodeCosineGeometry(const YAML::Node& /*params*/) {}

    [[nodiscard]] auto RInner    (double z) const -> double override;
    [[nodiscard]] auto ROuter    (double z) const -> double override;
    [[nodiscard]] auto DrInnerDz(double z) const -> double override;
    [[nodiscard]] auto DrOuterDz(double z) const -> double override;

    [[nodiscard]] auto Name() const -> std::string override { return "short_cathode_cosine"; }
};
