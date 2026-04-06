#pragma once
#include "iboundary_condition.hpp"
namespace YAML { class Node; }

/// Outflow (zero-gradient) boundary condition — L_HI face, iterates over m.
///
/// Copies all conservative variables from the last interior cell:
///   u_*[local_L][m] = u_*[local_L - 1][m]
///
/// Physical variables are reconstructed from u_* by the end-of-step call to
/// Fields::update_physical_from_u(), so no explicit physical copy is needed.
class OutflowBC : public IBoundaryCondition {
public:
    explicit OutflowBC(const YAML::Node& /*params*/) {}
    void        apply(BCContext& ctx) const override;
    std::string name()                const override { return "outflow"; }
};
