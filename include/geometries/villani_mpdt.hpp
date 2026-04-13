#pragma once
#include "igeometry.hpp"
namespace YAML { class Node; }

/// Coaxial MPDT geometry following the Villani (1982) benchmark setup.
///
/// Two coaxial cylindrical electrodes of *different* lengths define the
/// boundary profiles used to generate the body-fitted mesh:
///
///   Inner wall  =  cathode electrode  (axis of symmetry for z > z_cathode)
///   Outer wall  =  anode  electrode  (domain boundary  for z > z_anode)
///
/// Outer-wall (anode) profile
/// ──────────────────────────
///   z < z_anode − half_w_a          r_outer = r_anode
///   z_anode ± half_w_a   (blend)    cosine from r_anode → r_outer_domain
///   z > z_anode + half_w_a          r_outer = r_outer_domain
///
///   Setting half_w_a = 0 gives a sharp step (C⁰, not recommended for the
///   mesh-skew metric).  Any positive value yields a C¹ cosine transition.
///
/// Inner-wall (cathode) profile
/// ────────────────────────────
///   z < z_cathode − half_w_c        r_inner = r_cathode
///   z_cathode ± half_w_c  (blend)   cosine from r_cathode → r_inner_after
///   z > z_cathode + half_w_c        r_inner = r_inner_after
///
///   r_inner_after should be a small positive value that approximates the
///   axis of symmetry (0).  Setting it to exactly 0 causes a division-by-r
///   singularity in the solver; typical safe values are 1 % of r_cathode.
///
///   Setting half_w_c = 0 gives a sharp step at the cathode tip.
///
/// Reference dimensions (Villani 1982 / Tkachenko et al. 2023)
/// ─────────────────────────────────────────────────────────────
///   r_cathode       = 0.95 cm
///   r_anode         = 5.10 cm
///   r_outer_domain  = 10.2 cm
///   l_cathode       = 26.4 cm   →  z_cathode ≈ 0.518 (fraction of 51 cm domain)
///   l_anode         = 20.0 cm   →  z_anode   ≈ 0.392
///
/// All parameters are in the same dimensionless "simulation units" as the
/// rest of the geometry configuration.  The z parameters are normalised so
/// that z = 1 corresponds to the far end of the computational domain
/// (z = L_max × dz = 1 exactly).  Radial values share the same unit system.
///
/// ── YAML params (all optional, defaults reproduce the Villani setup) ────────
///
///   geometry:
///     type: villani_mpdt
///     params:
///       # Cathode (inner electrode)
///       r_cathode:              0.0186   # cathode radius  [sim units]
///       z_cathode:              0.518    # cathode tip z-position  [0, 1]
///       r_inner_after:          0.0002   # inner radius beyond cathode tip
///       cathode_tip_half_width: 0.015    # cosine blend half-width  (0 = sharp)
///
///       # Anode (outer electrode)
///       r_anode:                0.100    # anode radius
///       z_anode:                0.392    # anode tip z-position  [0, 1]
///       r_outer_domain:         0.200    # outer domain radius (beyond anode)
///       anode_tip_half_width:   0.015    # cosine blend half-width  (0 = sharp)
class VillaniMPDTGeometry : public IGeometry {
public:
    struct Params {
        // ── Cathode (inner electrode) ──────────────────────────────────────
        double r_cathode              = 0.0186;  ///< cathode radius (inner electrode)
        double z_cathode              = 0.518;   ///< normalised z-position of cathode tip
        double r_inner_after          = 0.0002;  ///< inner radius beyond cathode tip
        double cathode_tip_half_width = 0.015;   ///< cosine transition half-width (0 = sharp)

        // ── Anode (outer electrode) ────────────────────────────────────────
        double r_anode                = 0.100;   ///< anode radius (outer electrode)
        double z_anode                = 0.392;   ///< normalised z-position of anode tip
        double r_outer_domain         = 0.200;   ///< outer domain radius beyond the anode
        double anode_tip_half_width   = 0.015;   ///< cosine transition half-width (0 = sharp)
    };

    explicit VillaniMPDTGeometry(const YAML::Node& params);

    [[nodiscard]] auto RInner   (double z) const -> double override;
    [[nodiscard]] auto ROuter   (double z) const -> double override;
    [[nodiscard]] auto DrInnerDz(double z) const -> double override;
    [[nodiscard]] auto DrOuterDz(double z) const -> double override;

    [[nodiscard]] auto Name()      const -> std::string   override { return "villani_mpdt"; }
    [[nodiscard]] auto GetParams() const -> const Params& { return p_; }

private:
    Params p_;

    // ── Precomputed cathode-tip transition bounds ──────────────────────────
    double c_z_start_{};    ///< z_cathode − cathode_tip_half_width
    double c_z_end_{};      ///< z_cathode + cathode_tip_half_width
    double c_delta_r_{};    ///< r_inner_after − r_cathode  (negative: shrinking)
    double c_inv_width_{};  ///< 1 / (c_z_end_ − c_z_start_)  (0 when half_w = 0)

    // ── Precomputed anode-tip transition bounds ────────────────────────────
    double a_z_start_{};    ///< z_anode − anode_tip_half_width
    double a_z_end_{};      ///< z_anode + anode_tip_half_width
    double a_delta_r_{};    ///< r_outer_domain − r_anode  (positive: expanding)
    double a_inv_width_{};  ///< 1 / (a_z_end_ − a_z_start_)  (0 when half_w = 0)

    /// Evaluate a cosine-blended step from r_before to r_after in [z_s, z_e].
    /// Returns r_before for z < z_s and r_after for z > z_e.
    static auto CosineStep(double z, double z_s, double z_e,
                            double r_before, double delta_r,
                            double inv_w) noexcept -> double;

    /// Derivative of CosineStep with respect to z.
    static auto CosineStepDeriv(double z, double z_s, double z_e,
                                 double delta_r, double inv_w) noexcept -> double;
};
