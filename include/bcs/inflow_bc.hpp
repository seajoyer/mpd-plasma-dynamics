#pragma once
#include "iboundary_condition.hpp"
namespace YAML { class Node; }

/// Inflow (inlet) boundary condition — L_LO face, iterates over m.
///
/// Fixed quantities at l = 1 (global z = 0):
///   ρ = 1,  v_phi = 0,  v_r = 0
///   v_z  = u_2[2][m] / (ρ · r[1][m])   (taken from interior, mass-flux-matching)
///   H_z  = H_z0,  H_r = 0,  H_phi = r_0 / r[1][m]
///   e    = β / (2(γ-1)) · ρ^(γ-1)
///
/// Physical variables are set first; rebuild_u_from_physical() is called
/// at the end of each cell to keep u_* consistent.
class InflowBC : public IBoundaryCondition {
public:
    explicit InflowBC(const YAML::Node& /*params*/) {}
    void        apply(BCContext& ctx) const override;
    std::string name()                const override { return "inflow"; }
};
