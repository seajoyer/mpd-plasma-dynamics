#include "initial_condition_registry.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

#include "ics/expression_ic.hpp"

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
    -> std::unique_ptr<IInitialCondition>
{
    auto it = factories_.find(type);
    if (it == factories_.end()) {
        throw std::runtime_error(
            "InitialConditionRegistry: unknown IC type '" + type + "'.\n"
            "  The only built-in type is 'expression'.\n"
            "  Check for a typo in config.yaml under initial_conditions.type.");
    }
    return it->second(params);
}

// ============================================================
// Built-in IC registrations
// ============================================================

void RegisterAllInitialConditions() {
    InitialConditionRegistry& reg = InitialConditionRegistry::Instance();

    reg.Register("expression",
                 [](const YAML::Node& p) { return std::make_unique<ExpressionIC>(p); });

    // ----------------------------------------------------------------
    // To add a custom IC type:
    //   1. Subclass IInitialCondition (see include/iinitial_condition.hpp).
    //   2. Add a Register() call here following the pattern above.
    //   3. Reference the name in config.yaml under initial_conditions.type.
    // ----------------------------------------------------------------
}
