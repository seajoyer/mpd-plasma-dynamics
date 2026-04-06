#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace YAML { class Node; }
class IBoundaryCondition;

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

/// Register every built-in BC type.  Call once before constructing any Solver.
void register_all_bcs();
