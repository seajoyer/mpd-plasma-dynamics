#pragma once
#include "igeometry.hpp"
namespace YAML { class Node; }

/// Short cathode geometry with a cosine-smoothed inner-wall transition.
///
/// Outer wall: r_outer(z) = r_outer  (constant anode)
///
/// Inner wall: cosine blend from r_inner_before (cathode face) to
///             r_inner_after (thin inner electrode / axis):
///
///   z < z_start          r_inner = r_inner_before
///   z_start ≤ z < z_end  r_inner = r_inner_before
///                                 + (r_inner_after − r_inner_before)
///                                   · ½(1 − cos(π·ξ))
///   z ≥ z_end            r_inner = r_inner_after
///
/// where  z_start = z_center − transition_half_width,
///        z_end   = z_center + transition_half_width,
///        ξ = (z − z_start) / (z_end − z_start) ∈ [0, 1].
///
/// The derivative dr_inner/dz is C¹ continuous everywhere and zero outside
/// the transition band.
///
/// ── YAML params (all optional, defaults shown) ──────────────────────────────
///
///   geometry:
///     type: short_cathode_cosine
///     params:
///       z_center:               0.31    # axial centre of the cosine transition
///       transition_half_width:  0.015   # half-width of the transition band
///       r_inner_before:         0.200   # cathode-face inner radius
///       r_inner_after:          0.005   # thin-electrode inner radius
///       r_outer:                0.800   # anode (outer wall) radius
class ShortCathodeCosineGeometry : public IGeometry {
public:
    struct Params {
        double z_center              = 0.31;
        double transition_half_width = 0.015;
        double r_inner_before        = 0.200;
        double r_inner_after         = 0.005;
        double r_outer               = 0.800;
    };

    explicit ShortCathodeCosineGeometry(const YAML::Node& params);

    [[nodiscard]] auto RInner    (double z) const -> double override;
    [[nodiscard]] auto ROuter    (double z) const -> double override;
    [[nodiscard]] auto DrInnerDz (double z) const -> double override;
    [[nodiscard]] auto DrOuterDz (double z) const -> double override;

    [[nodiscard]] auto Name()      const -> std::string  override { return "short_cathode_cosine"; }
    [[nodiscard]] auto GetParams() const -> const Params& { return p_; }

private:
    Params p_;

    // Derived from p_ — computed once in the constructor.
    double z_start_{};    ///< p_.z_center − p_.transition_half_width
    double z_end_{};      ///< p_.z_center + p_.transition_half_width
    double delta_r_{};    ///< p_.r_inner_after − p_.r_inner_before
    double inv_width_{};  ///< 1 / (z_end_ − z_start_)
};
