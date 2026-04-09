#include "initial_condition_registry.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

#include "ics/uniform_mhd.hpp"

// ============================================================
// InitialConditionRegistry singleton
// ============================================================

auto InitialConditionRegistry::Instance() -> InitialConditionRegistry& {
    static InitialConditionRegistry inst;
    return inst;
}

void InitialConditionRegistry::Register(std::string name, Factory factory) {
    factories_.emplace(std::move(name), std::move(factory));
}

auto InitialConditionRegistry::Create(const std::string& type,
                                      const YAML::Node& params) const
    -> std::unique_ptr<IInitialCondition> {
    auto it = factories_.find(type);
    if (it == factories_.end()) {
        throw std::runtime_error(
            "InitialConditionRegistry: unknown IC type '" + type +
            "'.\n"
            "  Did you forget to call RegisterAllInitialConditions(), or "
            "mistype the name in config.yaml?");
    }
    return it->second(params);
}

// ============================================================
// Built-in IC registrations
// ============================================================

void RegisterAllInitialConditions() {
    InitialConditionRegistry& reg = InitialConditionRegistry::Instance();

    reg.Register("uniform_mhd",
                 [](const YAML::Node& p) { return std::make_unique<UniformMhdIC>(p); });

    // ----------------------------------------------------------------
    // Add new IC types here by following the pattern above.
    // The factory receives the YAML::Node from the `params` sub-node
    // in config.yaml (null node if the key is absent).
    //
    // Example:
    //   reg.Register("radial_profile", [](const YAML::Node& p) {
    //       return std::make_unique<RadialProfileIC>(p);
    //   });
    // ----------------------------------------------------------------
}
