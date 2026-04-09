#include "geometries/short_cathode_parabolic.hpp"

#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

// ── Constructor ──────────────────────────────────────────────────────────────

ShortCathodeParabolicGeometry::ShortCathodeParabolicGeometry(const YAML::Node& node)
{
    // Apply any YAML overrides on top of the default Params values.
    if (node && !node.IsNull()) {
        if (!node.IsMap()) {
            throw std::runtime_error(
                "ShortCathodeParabolicGeometry: 'params' must be a YAML map");
        }
        if (node["z_flat_end"])   p_.z_flat_end   = node["z_flat_end"].as<double>();
        if (node["z_throat"])     p_.z_throat     = node["z_throat"].as<double>();
        if (node["z_thin_start"]) p_.z_thin_start = node["z_thin_start"].as<double>();
        if (node["r_flat"])       p_.r_flat       = node["r_flat"].as<double>();
        if (node["r_thin"])       p_.r_thin       = node["r_thin"].as<double>();
        if (node["r_outer"])      p_.r_outer      = node["r_outer"].as<double>();
    }

    // Validate ordering and physical constraints.
    if (!(p_.z_flat_end < p_.z_throat && p_.z_throat < p_.z_thin_start)) {
        throw std::runtime_error(
            "ShortCathodeParabolicGeometry: requires "
            "z_flat_end < z_throat < z_thin_start");
    }
    if (p_.r_flat <= p_.r_thin) {
        throw std::runtime_error(
            "ShortCathodeParabolicGeometry: r_flat must be greater than r_thin");
    }
    if (p_.r_outer <= p_.r_flat) {
        throw std::runtime_error(
            "ShortCathodeParabolicGeometry: r_outer must be greater than r_flat");
    }

    // Derive the parabola coefficient and the arc centre so that:
    //   • the descending arc (z_flat_end → z_throat) starts at r_flat with
    //     zero slope and ends at the throat r_throat with zero slope,
    //   • the ascending arc (z_throat → z_thin_start) starts at r_throat
    //     with zero slope (same A → symmetric arcs sharing the throat),
    //   • the two arcs are C⁰ continuous at the throat.
    //
    // Both conditions are satisfied by:
    //   A             = r_flat / (2·Δz²)   where Δz = z_throat − z_flat_end
    //   z_arc_center  = 2·z_throat − z_flat_end   (mirror of z_flat_end)
    //
    // This gives r_throat = A·(z_throat − z_arc_center)² = A·Δz² = r_flat/2.
    const double dz  = p_.z_throat - p_.z_flat_end;
    A_           = p_.r_flat / (2.0 * dz * dz);
    z_arc_center_ = 2.0 * p_.z_throat - p_.z_flat_end;
}

// ── IGeometry interface ──────────────────────────────────────────────────────

auto ShortCathodeParabolicGeometry::RInner(double z) const -> double {
    if (z < p_.z_flat_end) {
        return p_.r_flat;
    }
    if (z < p_.z_throat) {
        const double dz = z - p_.z_flat_end;
        return p_.r_flat - A_ * dz * dz;
    }
    if (z < p_.z_thin_start) {
        const double dz = z - z_arc_center_;
        return A_ * dz * dz;
    }
    return p_.r_thin;
}

auto ShortCathodeParabolicGeometry::ROuter(double /*z*/) const -> double {
    return p_.r_outer;
}

auto ShortCathodeParabolicGeometry::DrInnerDz(double z) const -> double {
    if (z < p_.z_flat_end || z >= p_.z_thin_start) {
        return 0.0;
    }
    if (z < p_.z_throat) {
        return -2.0 * A_ * (z - p_.z_flat_end);
    }
    // z_throat ≤ z < z_thin_start
    return  2.0 * A_ * (z - z_arc_center_);
}

auto ShortCathodeParabolicGeometry::DrOuterDz(double /*z*/) const -> double {
    return 0.0;
}
