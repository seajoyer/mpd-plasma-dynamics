#include "bc_registry.hpp"

#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "geometries/coaxial_nozzle.hpp"

// ============================================================
// GeometryRegistry singleton
// ============================================================

GeometryRegistry& GeometryRegistry::instance() {
    static GeometryRegistry inst;
    return inst;
}

void GeometryRegistry::register_geometry(std::string name, Factory factory) {
    factories_.emplace(std::move(name), std::move(factory));
}

std::unique_ptr<IGeometry>
GeometryRegistry::create(const std::string& type, const YAML::Node& params) const {
    auto it = factories_.find(type);
    if (it == factories_.end())
        throw std::runtime_error("GeometryRegistry: unknown geometry type '" + type + "'");
    return it->second(params);
}

// ============================================================
// Built-in geometry registrations
// ============================================================

void register_all_geometries() {
    GeometryRegistry& reg = GeometryRegistry::instance();

    reg.register_geometry("coaxial_nozzle", [](const YAML::Node& p) {
        return std::make_unique<CoaxialNozzleGeometry>(p);
    });

    // ----------------------------------------------------------------
    // Add new geometry types here by following the pattern above.
    // The factory receives the YAML::Node from the `params` sub-node
    // in config.yaml (null node if the key is absent).
    //
    // Example:
    //   reg.register_geometry("straight_channel", [](const YAML::Node& p) {
    //       return std::make_unique<StraightChannelGeometry>(p);
    //   });
    // ----------------------------------------------------------------
}
