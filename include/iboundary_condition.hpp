#pragma once

#include <string>
#include "bc_context.hpp"

/// Abstract interface for a single boundary-condition kernel.
///
/// Implementations are stateless after construction: all mutable simulation
/// state is accessed through the BCContext passed to Apply().  This means
/// implementations are trivially thread-safe and testable without MPI.
///
/// The sole built-in implementation is PerFieldBC, which covers all general
/// Dirichlet / Neumann / WallTangent / AxisLF cases.  A custom implementation
/// can be added by:
///
///   1. Subclassing IBoundaryCondition and implementing Apply() and Name().
///   2. Creating an instance in FaceBC::FromConfig() based on a new key
///      in BCSegmentConfig.
///
/// Ownership / face convention
/// ────────────────────────────
/// Each concrete implementation is associated with exactly one face type
/// (L_LO, L_HI, M_LO, or M_HI) and stores the fixed axis index internally.
class IBoundaryCondition {
public:
    virtual ~IBoundaryCondition() = default;

    /// Apply the boundary condition to the range [ctx.local_lo, ctx.local_hi].
    virtual void Apply(BCContext& ctx) const = 0;

    /// Short identifier for log messages.
    [[nodiscard]] virtual auto Name() const -> std::string = 0;
};
