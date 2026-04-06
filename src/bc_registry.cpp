#include "bc_registry.hpp"

#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "bcs/inflow_bc.hpp"
#include "bcs/outflow_bc.hpp"
#include "bcs/outer_wall_bc.hpp"
#include "bcs/solid_wall_bc.hpp"
#include "bcs/axis_symmetry_bc.hpp"

// ============================================================
// BCRegistry singleton
// ============================================================

BCRegistry& BCRegistry::instance() {
    static BCRegistry inst;
    return inst;
}

void BCRegistry::register_bc(std::string name, Factory factory) {
    factories_.emplace(std::move(name), std::move(factory));
}

std::unique_ptr<IBoundaryCondition>
BCRegistry::create(const std::string& type, const YAML::Node& params) const {
    auto it = factories_.find(type);
    if (it == factories_.end())
        throw std::runtime_error("BCRegistry: unknown BC type '" + type + "'.\n"
            "  Did you forget to call register_all_bcs(), or mistype the name in config.yaml?");
    return it->second(params);
}

// ============================================================
// Built-in BC registrations
// ============================================================

void register_all_bcs() {
    BCRegistry& reg = BCRegistry::instance();

    reg.register_bc("inflow",        [](const YAML::Node& p) {
        return std::make_unique<InflowBC>(p);
    });
    reg.register_bc("outflow",       [](const YAML::Node& p) {
        return std::make_unique<OutflowBC>(p);
    });
    reg.register_bc("outer_wall",    [](const YAML::Node& p) {
        return std::make_unique<OuterWallBC>(p);
    });
    reg.register_bc("solid_wall",    [](const YAML::Node& p) {
        return std::make_unique<SolidWallBC>(p);
    });
    reg.register_bc("axis_symmetry", [](const YAML::Node& p) {
        return std::make_unique<AxisSymmetryBC>(p);
    });

    // ----------------------------------------------------------------
    // Add new BC types here following the pattern above.
    //
    // The factory lambda receives the YAML::Node from the `params`
    // sub-node for that segment (null node if the key is absent in
    // config.yaml).  Assign it to a member in the constructor:
    //
    //   reg.register_bc("my_inlet", [](const YAML::Node& p) {
    //       return std::make_unique<MyInletBC>(p);
    //   });
    // ----------------------------------------------------------------
}
