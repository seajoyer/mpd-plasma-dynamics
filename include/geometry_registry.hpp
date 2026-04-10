#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace YAML { class Node; }
class IGeometry;

/// Singleton factory registry for IGeometry implementations.
///
/// Populated at startup by register_all_geometries()
/// (see src/geometry_registry.cpp).
/// Looked up in main() when constructing the Grid.
class GeometryRegistry {
public:
    using Factory = std::function<std::unique_ptr<IGeometry>(const YAML::Node&)>;

    static auto Instance() -> GeometryRegistry&;

    void RegisterGeometry(std::string name, Factory factory);

    /// Throws std::runtime_error if `type` is unknown.
    auto Create(const std::string& type,
                                       const YAML::Node& params) const -> std::unique_ptr<IGeometry>;

private:
    GeometryRegistry() = default;
    std::unordered_map<std::string, Factory> factories_;
};

/// Register every built-in geometry type.  Call once before constructing any Grid.
void RegisterAllGeometries();
