#pragma once
#include "iboundary_condition.hpp"
namespace YAML { class Node; }

/// Axis-of-symmetry boundary condition — M_LO face segment, iterates over l.
///
/// Applies a one-sided Lax–Friedrichs update at m = 1 that enforces the
/// axis-of-symmetry conditions:
///   v_r = 0,  v_phi = 0,  H_phi = 0,  H_r = 0
///
/// The non-zero conservative variables (u_1, u_2, u_5, u_7) are updated
/// with a half-stencil in the m-direction: the "inner" neighbour at m = 0
/// is treated as a mirror of m = 1 via the symmetry condition.
///
/// u_* is set directly (not via physical variables); the end-of-step call
/// to Fields::update_physical_from_u() reconstructs physical variables.
///
/// Typically applied for the open-axis region (global l > L_end).
class AxisSymmetryBC : public IBoundaryCondition {
public:
    explicit AxisSymmetryBC(const YAML::Node& /*params*/) {}
    void        apply(BCContext& ctx) const override;
    std::string name()                const override { return "axis_symmetry"; }
};
