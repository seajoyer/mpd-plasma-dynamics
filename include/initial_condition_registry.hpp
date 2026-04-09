#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "iinitial_condition.hpp"

namespace YAML {
class Node;
}
class IInitialCondition;

/// Singleton factory registry for IInitialCondition implementations.
///
/// Populated at startup by RegisterAllInitialConditions()
/// (see src/initial_condition_registry.cpp).
/// Looked up in main() to construct the initial field state.
class InitialConditionRegistry {
   public:
    using Factory = std::function<std::unique_ptr<IInitialCondition>(const YAML::Node&)>;

    static auto Instance() -> InitialConditionRegistry&;

    void Register(std::string name, Factory factory);

    /// Throws std::runtime_error if `type` is unknown.
    [[nodiscard]] auto Create(const std::string& type, const YAML::Node& params) const
        -> std::unique_ptr<IInitialCondition>;

   private:
    InitialConditionRegistry() = default;
    std::unordered_map<std::string, Factory> factories_;
};

/// Register every built-in IC type.  Call once before constructing Fields.
void RegisterAllInitialConditions();
