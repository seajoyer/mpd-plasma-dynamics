#pragma once
#include "iboundary_condition.hpp"
namespace YAML { class Node; }

/// Solid inner-wall boundary condition — M_LO face segment, iterates over l.
///
/// Wall-tangent condition at m = 1: copies scalar fields from the first
/// interior cell (m = 2) and derives the radial velocity from the wall slope:
///
///   ρ, v_z, v_phi, e, H_phi, H_z  copied from m+1
///   v_r  = v_z[l][m+1] · r_z[l][m+1]   (tangential flow condition)
///   H_r  = H_z[l][m+1] · r_z[l][m+1]
///
/// Typically used for the nozzle/electrode region (global l ∈ [0, L_end]).
class SolidWallBC : public IBoundaryCondition {
public:
    explicit SolidWallBC(const YAML::Node& /*params*/) {}
    void        apply(BCContext& ctx) const override;
    std::string name()                const override { return "solid_wall"; }
};
