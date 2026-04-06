#pragma once

#include <string>
#include "bc_context.hpp"

/// Abstract interface for a single boundary-condition kernel.
///
/// Implementations are stateless after construction: all mutable simulation
/// state is accessed through the BCContext passed to apply().  This means
/// implementations are trivially thread-safe and testable without MPI.
///
/// Ownership / face convention
/// ────────────────────────────
/// Each concrete implementation is designed for exactly one face type
/// (L_LO, L_HI, M_LO, or M_HI).  The fixed index is implicit:
///
///   Typical BC type  │ Face   │ fixed index used inside apply()
///   ─────────────────┼────────┼──────────────────────────────────
///   InflowBC         │ L_LO   │ l = 1
///   OutflowBC        │ L_HI   │ l = ctx.mpi.local_L
///   OuterWallBC      │ M_HI   │ m = ctx.mpi.local_M
///   SolidWallBC      │ M_LO   │ m = 1
///   AxisSymmetryBC   │ M_LO   │ m = 1
///
/// The apply() call is issued only on the rank that owns the face.  BCs do
/// not need to guard with is_*_boundary() themselves.
///
/// Registering a new BC
/// ─────────────────────
/// Add the implementation files, then register the factory inside
/// register_all_bcs() in src/bc_registry.cpp:
///
///   reg.register_bc("my_bc", [](const YAML::Node& p) {
///       return std::make_unique<MyBC>(p);
///   });
///
/// Then use the name in config.yaml under boundary_conditions.<face>.
class IBoundaryCondition {
public:
    virtual ~IBoundaryCondition() = default;

    /// Apply the boundary condition to the range [ctx.local_lo, ctx.local_hi].
    ///
    /// Implementations typically contain one `#pragma omp parallel for` loop
    /// iterating over that range, setting physical variables and calling
    /// rebuild_u_from_physical() or setting u_* directly.
    virtual void apply(BCContext& ctx) const = 0;

    /// Short identifier for log messages (matches the registered name).
    virtual std::string name() const = 0;
};
