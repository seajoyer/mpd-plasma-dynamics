#include "geometries/short_cathode_cosine.hpp"

#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

// ── Constructor ──────────────────────────────────────────────────────────────

ShortCathodeCosineGeometry::ShortCathodeCosineGeometry(const YAML::Node& node)
{
    // Apply any YAML overrides on top of the default Params values.
    if (node && !node.IsNull()) {
        if (!node.IsMap()) {
            throw std::runtime_error(
                "ShortCathodeCosineGeometry: 'params' must be a YAML map");
        }
        if (node["z_center"])              p_.z_center              = node["z_center"].as<double>();
        if (node["transition_half_width"]) p_.transition_half_width = node["transition_half_width"].as<double>();
        if (node["r_inner_before"])        p_.r_inner_before        = node["r_inner_before"].as<double>();
        if (node["r_inner_after"])         p_.r_inner_after         = node["r_inner_after"].as<double>();
        if (node["r_outer"])               p_.r_outer               = node["r_outer"].as<double>();
    }

    // Validate
    if (p_.transition_half_width <= 0.0) {
        throw std::runtime_error(
            "ShortCathodeCosineGeometry: transition_half_width must be positive");
    }
    if (p_.r_outer <= p_.r_inner_before || p_.r_outer <= p_.r_inner_after) {
        throw std::runtime_error(
            "ShortCathodeCosineGeometry: r_outer must be greater than both "
            "r_inner_before and r_inner_after");
    }

    // Precompute derived constants.
    z_start_   = p_.z_center - p_.transition_half_width;
    z_end_     = p_.z_center + p_.transition_half_width;
    delta_r_   = p_.r_inner_after - p_.r_inner_before;
    inv_width_ = 1.0 / (z_end_ - z_start_);
}

// ── IGeometry interface ──────────────────────────────────────────────────────

auto ShortCathodeCosineGeometry::RInner(double z) const -> double {
    if (z < z_start_) {
        return p_.r_inner_before;
    }
    if (z < z_end_) {
        const double xi = (z - z_start_) * inv_width_;
        return p_.r_inner_before + delta_r_ * 0.5 * (1.0 - std::cos(M_PI * xi));
    }
    return p_.r_inner_after;
}

auto ShortCathodeCosineGeometry::ROuter(double /*z*/) const -> double {
    return p_.r_outer;
}

auto ShortCathodeCosineGeometry::DrInnerDz(double z) const -> double {
    if (z < z_start_ || z >= z_end_) {
        return 0.0;
    }
    // d/dz [ ½(1 − cos(π·ξ)) ] = ½·π·sin(π·ξ) · dξ/dz,   dξ/dz = inv_width_
    const double xi = (z - z_start_) * inv_width_;
    return delta_r_ * 0.5 * M_PI * std::sin(M_PI * xi) * inv_width_;
}

auto ShortCathodeCosineGeometry::DrOuterDz(double /*z*/) const -> double {
    return 0.0;
}
