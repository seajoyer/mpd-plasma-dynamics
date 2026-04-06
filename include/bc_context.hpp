#pragma once

#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// All data an IBoundaryCondition implementation might need.
///
/// Constructed cheaply (reference members only) by FaceBC::apply() for each
/// active segment and passed by reference to IBoundaryCondition::apply().
///
/// Index semantics for local_lo / local_hi
/// ─────────────────────────────────────────
/// Each face has one *fixed* index and one *free* axis that the BC iterates.
///
///   Face  │ fixed index      │ free axis  │ local_lo / local_hi
///   ──────┼──────────────────┼────────────┼─────────────────────
///   L_LO  │ l = 1            │ m          │ m-index range (local)
///   L_HI  │ l = local_L      │ m          │ m-index range (local)
///   M_LO  │ m = 1            │ l          │ l-index range (local)
///   M_HI  │ m = local_M      │ l          │ l-index range (local)
///
/// The range [local_lo, local_hi] is inclusive and has already been clipped
/// to the cells this MPI rank owns.  BC implementations must iterate exactly
/// over this range and not touch any cell outside it.
struct BCContext {
    Fields&           fields;
    const Grid&       grid;
    const SimConfig&  cfg;
    const MPIManager& mpi;
    double            dt;
    int               local_lo;  ///< first local free-axis index (inclusive)
    int               local_hi;  ///< last  local free-axis index (inclusive)
};
