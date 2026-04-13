#include "geometries/villani_mpdt.hpp"

#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

// ── Constructor ──────────────────────────────────────────────────────────────

VillaniMPDTGeometry::VillaniMPDTGeometry(const YAML::Node& node)
{
    if (node && !node.IsNull()) {
        if (!node.IsMap()) {
            throw std::runtime_error(
                "VillaniMPDTGeometry: 'params' must be a YAML map");
        }
        if (node["r_cathode"])              p_.r_cathode              = node["r_cathode"].as<double>();
        if (node["z_cathode"])              p_.z_cathode              = node["z_cathode"].as<double>();
        if (node["r_inner_after"])          p_.r_inner_after          = node["r_inner_after"].as<double>();
        if (node["cathode_tip_half_width"]) p_.cathode_tip_half_width = node["cathode_tip_half_width"].as<double>();

        if (node["r_anode"])                p_.r_anode                = node["r_anode"].as<double>();
        if (node["z_anode"])                p_.z_anode                = node["z_anode"].as<double>();
        if (node["r_outer_domain"])         p_.r_outer_domain         = node["r_outer_domain"].as<double>();
        if (node["anode_tip_half_width"])   p_.anode_tip_half_width   = node["anode_tip_half_width"].as<double>();
    }

    // ── Validate ──────────────────────────────────────────────────────────
    if (p_.r_cathode <= 0.0)
        throw std::runtime_error("VillaniMPDTGeometry: r_cathode must be positive");

    if (p_.r_inner_after <= 0.0)
        throw std::runtime_error(
            "VillaniMPDTGeometry: r_inner_after must be positive (> 0) to avoid "
            "a 1/r singularity on the axis.  Use a small value such as 1 % of r_cathode.");

    if (p_.r_inner_after >= p_.r_cathode)
        throw std::runtime_error(
            "VillaniMPDTGeometry: r_inner_after must be less than r_cathode "
            "(the inner radius shrinks toward the axis beyond the cathode tip)");

    if (p_.r_anode <= p_.r_cathode)
        throw std::runtime_error(
            "VillaniMPDTGeometry: r_anode must be greater than r_cathode");

    if (p_.r_outer_domain <= p_.r_anode)
        throw std::runtime_error(
            "VillaniMPDTGeometry: r_outer_domain must be greater than r_anode");

    if (p_.z_anode <= 0.0 || p_.z_anode >= 1.0)
        throw std::runtime_error(
            "VillaniMPDTGeometry: z_anode must be in the open interval (0, 1)");

    if (p_.z_cathode <= p_.z_anode)
        throw std::runtime_error(
            "VillaniMPDTGeometry: z_cathode must be greater than z_anode "
            "(the cathode is longer than the anode in the Villani setup)");

    if (p_.z_cathode >= 1.0)
        throw std::runtime_error(
            "VillaniMPDTGeometry: z_cathode must be less than 1 "
            "(the cathode tip must be inside the computational domain)");

    if (p_.cathode_tip_half_width < 0.0)
        throw std::runtime_error(
            "VillaniMPDTGeometry: cathode_tip_half_width must be >= 0");

    if (p_.anode_tip_half_width < 0.0)
        throw std::runtime_error(
            "VillaniMPDTGeometry: anode_tip_half_width must be >= 0");

    // ── Precompute transition constants ───────────────────────────────────

    // Cathode tip transition  (r shrinks from r_cathode → r_inner_after)
    c_z_start_  = p_.z_cathode - p_.cathode_tip_half_width;
    c_z_end_    = p_.z_cathode + p_.cathode_tip_half_width;
    c_delta_r_  = p_.r_inner_after - p_.r_cathode;   // always negative
    c_inv_width_ = (p_.cathode_tip_half_width > 0.0)
                       ? 1.0 / (c_z_end_ - c_z_start_)
                       : 0.0;

    // Anode tip transition  (r grows from r_anode → r_outer_domain)
    a_z_start_  = p_.z_anode - p_.anode_tip_half_width;
    a_z_end_    = p_.z_anode + p_.anode_tip_half_width;
    a_delta_r_  = p_.r_outer_domain - p_.r_anode;    // always positive
    a_inv_width_ = (p_.anode_tip_half_width > 0.0)
                       ? 1.0 / (a_z_end_ - a_z_start_)
                       : 0.0;
}

// ── Static helpers ───────────────────────────────────────────────────────────

auto VillaniMPDTGeometry::CosineStep(double z, double z_s, double z_e,
                                      double r_before, double delta_r,
                                      double inv_w) noexcept -> double
{
    if (z < z_s) return r_before;
    if (z >= z_e) return r_before + delta_r;
    // Cosine blend: ½(1 − cos(π·ξ)) where ξ ∈ [0, 1]
    const double xi = (z - z_s) * inv_w;
    return r_before + delta_r * 0.5 * (1.0 - std::cos(M_PI * xi));
}

auto VillaniMPDTGeometry::CosineStepDeriv(double z, double z_s, double z_e,
                                           double delta_r, double inv_w) noexcept -> double
{
    if (z < z_s || z >= z_e) return 0.0;
    // d/dz [ ½(1 − cos(π·ξ)) ] = ½·π·sin(π·ξ) · (1/width)
    const double xi = (z - z_s) * inv_w;
    return delta_r * 0.5 * M_PI * std::sin(M_PI * xi) * inv_w;
}

// ── IGeometry interface ──────────────────────────────────────────────────────

auto VillaniMPDTGeometry::RInner(double z) const -> double
{
    if (p_.cathode_tip_half_width <= 0.0) {
        // Sharp step: cathode radius up to z_cathode, then r_inner_after
        return (z < p_.z_cathode) ? p_.r_cathode : p_.r_inner_after;
    }
    return CosineStep(z, c_z_start_, c_z_end_,
                      p_.r_cathode, c_delta_r_, c_inv_width_);
}

auto VillaniMPDTGeometry::ROuter(double z) const -> double
{
    if (p_.anode_tip_half_width <= 0.0) {
        // Sharp step: anode radius up to z_anode, then r_outer_domain
        return (z < p_.z_anode) ? p_.r_anode : p_.r_outer_domain;
    }
    return CosineStep(z, a_z_start_, a_z_end_,
                      p_.r_anode, a_delta_r_, a_inv_width_);
}

auto VillaniMPDTGeometry::DrInnerDz(double z) const -> double
{
    if (p_.cathode_tip_half_width <= 0.0) return 0.0;
    return CosineStepDeriv(z, c_z_start_, c_z_end_, c_delta_r_, c_inv_width_);
}

auto VillaniMPDTGeometry::DrOuterDz(double z) const -> double
{
    if (p_.anode_tip_half_width <= 0.0) return 0.0;
    return CosineStepDeriv(z, a_z_start_, a_z_end_, a_delta_r_, a_inv_width_);
}
