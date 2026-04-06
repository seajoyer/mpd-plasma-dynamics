#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

// Forward declarations to avoid including heavy yaml-cpp here.
namespace YAML { class Node; }
class IBoundaryCondition;
class IGeometry;

// ============================================================
// BCRegistry
// ============================================================

/// Singleton factory registry for IBoundaryCondition implementations.
///
/// Populated at startup by register_all_bcs() (see src/bc_registry.cpp).
/// Looked up at solver-construction time when FaceBC::from_config() turns
/// BCFaceConfig segments into live IBoundaryCondition objects.
class BCRegistry {
public:
    using Factory = std::function<std::unique_ptr<IBoundaryCondition>(const YAML::Node&)>;

    static BCRegistry& instance();

    void register_bc(std::string name, Factory factory);

    /// Throws std::runtime_error if `type` is unknown.
    std::unique_ptr<IBoundaryCondition> create(const std::string& type,
                                                const YAML::Node& params) const;

private:
    BCRegistry() = default;
    std::unordered_map<std::string, Factory> factories_;
};

// ============================================================
// GeometryRegistry
// ============================================================

/// Singleton factory registry for IGeometry implementations.
///
/// Populated at startup by register_all_geometries() (see src/geometry_registry.cpp).
class GeometryRegistry {
public:
    using Factory = std::function<std::unique_ptr<IGeometry>(const YAML::Node&)>;

    static GeometryRegistry& instance();

    void register_geometry(std::string name, Factory factory);

    /// Throws std::runtime_error if `type` is unknown.
    std::unique_ptr<IGeometry> create(const std::string& type,
                                       const YAML::Node& params) const;

private:
    GeometryRegistry() = default;
    std::unordered_map<std::string, Factory> factories_;
};

// ============================================================
// Startup registration functions
// ============================================================

/// Register every built-in BC type.  Call once before constructing any Solver.
void register_all_bcs();

/// Register every built-in geometry type.  Call once before constructing any Grid.
void register_all_geometries();
