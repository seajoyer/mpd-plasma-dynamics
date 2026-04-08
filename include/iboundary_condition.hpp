#pragma once

#include <string>
#include "bc_context.hpp"

/// Abstract interface for a single boundary-condition kernel.
///
/// Implementations are stateless after construction: all mutable simulation
/// state is accessed through the BCContext passed to apply().  This means
/// implementations are trivially thread-safe and testable without MPI.
///
/// The sole built-in implementation is PerFieldBC, which covers all general
/// Dirichlet / Neumann / WallTangent / AxisLF cases.  A custom implementation
/// can be added by:
///
///   1. Subclassing IBoundaryCondition and implementing apply() and name().
///   2. Creating an instance in FaceBC::from_config() based on a new key
///      in BCSegmentConfig.
///
/// Ownership / face convention
/// ────────────────────────────
/// Each concrete implementation is associated with exactly one face type
/// (L_LO, L_HI, M_LO, or M_HI) and stores the fixed axis index internally.
/// The apply() call is issued only on the rank that owns the face; BCs do
/// not need to guard with is_*_boundary() themselves.
class IBoundaryCondition {
public:
    virtual ~IBoundaryCondition() = default;

    /// Apply the boundary condition to the range [ctx.local_lo, ctx.local_hi].
    virtual void Apply(BCContext& ctx) const = 0;

    /// Short identifier for log messages.
    [[nodiscard]] virtual auto Name() const -> std::string = 0;
};
