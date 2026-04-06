#pragma once
#include "iboundary_condition.hpp"
namespace YAML { class Node; }

/// Outer-wall boundary condition — M_HI face, iterates over l.
///
/// Wall-tangent condition: copies scalar fields from the last interior cell
/// (m = local_M - 1) and derives the radial velocity from the wall slope:
///
///   ρ, v_z, v_phi, e, H_phi, H_z  copied from m-1
///   v_r  = v_z · r_z[l][local_M]   (tangential flow condition)
///   H_r  = H_z · r_z[l][local_M]
class OuterWallBC : public IBoundaryCondition {
public:
    explicit OuterWallBC(const YAML::Node& /*params*/) {}
    void        apply(BCContext& ctx) const override;
    std::string name()                const override { return "outer_wall"; }
};
