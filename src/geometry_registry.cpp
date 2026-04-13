#include "geometry_registry.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

#include "geometries/short_cathode_cosine.hpp"
#include "geometries/short_cathode_parabolic.hpp"
#include "geometries/villani_mpdt.hpp"

// ============================================================
// GeometryRegistry singleton
// ============================================================

auto GeometryRegistry::Instance() -> GeometryRegistry& {
    static GeometryRegistry inst;
    return inst;
}

void GeometryRegistry::RegisterGeometry(std::string name, Factory factory) {
    factories_.emplace(std::move(name), std::move(factory));
}

auto GeometryRegistry::Create(const std::string& type, const YAML::Node& params) const
    -> std::unique_ptr<IGeometry> {
    auto it = factories_.find(type);
    if (it == factories_.end()) {
        throw std::runtime_error("GeometryRegistry: unknown geometry type '" + type +
                                 "'.\n"
                                 "  Did you forget to call register_all_geometries(), or "
                                 "mistype the name in config.yaml?");
    }
    return it->second(params);
}

// ============================================================
// Built-in geometry registrations
// ============================================================

void RegisterAllGeometries() {
    GeometryRegistry& reg = GeometryRegistry::Instance();

    reg.RegisterGeometry("short_cathode_parabolic", [](const YAML::Node& p) {
        return std::make_unique<ShortCathodeParabolicGeometry>(p);
    });

    reg.RegisterGeometry("short_cathode_cosine", [](const YAML::Node& p) {
        return std::make_unique<ShortCathodeCosineGeometry>(p);
    });

    reg.RegisterGeometry("villani_mpdt", [](const YAML::Node& p) {
        return std::make_unique<VillaniMPDTGeometry>(p);
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
