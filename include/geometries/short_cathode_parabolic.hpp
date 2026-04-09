#pragma once
#include "igeometry.hpp"
namespace YAML { class Node; }

/// Short cathode geometry with a parabolic-smoothed inner-wall transition.
///
/// Outer wall: r_outer(z) = r_outer  (constant)
///
/// Inner wall: four-piece piecewise profile with a throat:
///
///   z < z_flat_end
///       r_inner = r_flat
///
///   z_flat_end ≤ z < z_throat
///       r_inner = r_flat − A·(z − z_flat_end)²
///       (parabola descending from r_flat, tangent to the rising arc below)
///
///   z_throat ≤ z < z_thin_start
///       r_inner = A·(z − z_arc_center)²
///       (parabola rising from the throat toward z_arc_center)
///
///   z ≥ z_thin_start
///       r_inner = r_thin
///
/// The coefficient A and the arc-centre z_arc_center are derived
/// automatically from the four free parameters so that r and dr/dz
/// are C⁰ continuous at z_flat_end and C⁰ continuous at z_throat.
/// (The two arcs share the same A, guaranteeing C¹ continuity at the
/// throat where both derivatives are zero by symmetry.)
///
/// Derivation:
///   Continuity of r at z_throat:
///     r_flat − A·(z_throat − z_flat_end)² = A·(z_throat − z_arc_center)²
///   Zero slope at the throat (both parabolas have extrema there — this
///   constrains z_arc_center to be chosen so both arcs touch the same r
///   at z_throat with zero derivative):
///     A = r_flat / (2·(z_throat − z_flat_end)²)
///     z_arc_center = 2·z_throat − z_flat_end
///       (mirror of z_flat_end about z_throat)
///
/// ── YAML params (all optional, defaults shown) ──────────────────────────────
///
///   geometry:
///     type: short_cathode_parabolic
///     params:
///       z_flat_end:    0.3     # end of the flat cathode section
///       z_throat:      0.4     # axial position of the inner-wall throat
///       z_thin_start:  0.478   # start of the thin inner-electrode section
///       r_flat:        0.200   # cathode-face inner radius (flat section)
///       r_thin:        0.005   # thin-electrode inner radius
///       r_outer:       0.800   # anode (outer wall) radius
class ShortCathodeParabolicGeometry : public IGeometry {
public:
    struct Params {
        double z_flat_end   = 0.3;
        double z_throat     = 0.4;
        double z_thin_start = 0.478;
        double r_flat       = 0.200;
        double r_thin       = 0.005;
        double r_outer      = 0.800;
    };

    explicit ShortCathodeParabolicGeometry(const YAML::Node& params);

    [[nodiscard]] auto RInner    (double z) const -> double override;
    [[nodiscard]] auto ROuter    (double z) const -> double override;
    [[nodiscard]] auto DrInnerDz (double z) const -> double override;
    [[nodiscard]] auto DrOuterDz (double z) const -> double override;

    [[nodiscard]] auto Name()      const -> std::string   override { return "short_cathode_parabolic"; }
    [[nodiscard]] auto GetParams() const -> const Params& { return p_; }

private:
    Params p_;

    // Derived from p_ — computed once in the constructor.
    double A_{};             ///< parabola coefficient
    double z_arc_center_{};  ///< centre of the rising arc  (= 2·z_throat − z_flat_end)
};
