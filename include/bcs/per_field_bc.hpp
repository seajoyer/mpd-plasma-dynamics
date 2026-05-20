#pragma once

#include <memory>
#include <string>
#include <vector>

#include "iboundary_condition.hpp"
#include "config.hpp"
#include "face_bc.hpp"

/// Generic boundary condition that applies an independent FieldCond to each
/// of the eight physical variables (rho, v_z, v_r, v_phi, e, H_z, H_r, H_phi)
/// at a boundary segment.
///
/// Supported condition types (see FieldCondType for full documentation):
///
///   Neumann      Zero-gradient copy from the interior neighbour, with an
///                optional constant offset: f_bc = f_nb + offset.
///
///   Dirichlet    Fix to a constant value.
///
///   Expression   Evaluate a user-supplied mathematical expression at each
///                boundary cell.  Expressions are compiled once at
///                construction time via exprtk.  All physical fields,
///                neighbour values, grid quantities, and physics constants
///                are available as named symbols — see FieldCondType::Expression
///                documentation for the full symbol table.
///
///                Cross-field references work within a single cell evaluation:
///                a field can reference any field earlier in the evaluation
///                order (rho → v_z → v_r → v_phi → H_z → H_r → H_phi → e).
///
///   AxisLF       Half-stencil Lax–Friedrichs update for the axis of symmetry.
///                Valid only for rho / v_z / e / H_z on the M_LO face.
///
///                Time-level consistency: the AxisLF helpers derive every
///                physical quantity they need directly from the conservative
///                arrays f.u0_*, which carry the time-n state throughout
///                the entire Solver::Advance() call.
///
/// Step flow for each boundary cell
/// ─────────────────────────────────
///   1. Evaluate Neumann / Dirichlet / Expression conditions for each field
///      in order, updating the expression cross-reference variables as we go.
///   2. Call RebuildUFromPhysical() to keep conservative arrays consistent.
///   3. Overwrite u_* for any AxisLF fields with the one-sided LF stencil.
///
/// Step 3 is a no-op when no AxisLF conditions are present.
///
/// Thread safety
/// ─────────────
/// exprtk symbol tables hold mutable per-cell state and are not thread-safe
/// when shared.  To keep the boundary loop parallel even when expressions
/// are in use, PerFieldBC maintains one private ExprImpl instance per
/// OpenMP thread (sized to omp_get_max_threads() at construction).  The
/// parallel loop in Apply() indexes into this pool via omp_get_thread_num(),
/// so each thread evaluates expressions against its own symbol table.
///
/// Per-thread state preserves the documented cross-field reference semantics:
/// the rho → v_z → … → e ordering is *within* a single cell, and each
/// thread evaluates one cell start-to-finish before moving to the next.
class PerFieldBC : public IBoundaryCondition {
public:
    /// @param face  Which of the four Cartesian faces this segment belongs to.
    /// @param seg   Per-field condition configuration read from config.yaml.
    PerFieldBC(enum FaceBC::Face face, const BCSegmentConfig& seg);

    /// Defined in the .cpp where ExprImpl is a complete type.
    ~PerFieldBC() override;

    void        Apply(BCContext& ctx) const override;
    [[nodiscard]] auto Name() const -> std::string override { return "per_field"; }

private:
    enum FaceBC::Face face_;

    FieldCond rho_;
    FieldCond v_z_;
    FieldCond v_r_;
    FieldCond v_phi_;
    FieldCond e_;
    FieldCond H_z_;
    FieldCond H_r_;
    FieldCond H_phi_;

    bool has_axis_lf_;      ///< true if any field uses AxisLF
    bool has_expressions_;  ///< true if any field uses Expression

    // ---- Per-thread expression engines (empty when has_expressions_ == false) ----
    struct ExprImpl;
    std::vector<std::unique_ptr<ExprImpl>> expr_pool_;

    // ---- AxisLF stencil helpers (M_LO face, m = 1) ----
    //
    // All four helpers read time-n state from f.u0_*; gamma is passed in
    // because AxisLfU2 / AxisLfU5 need it to reconstruct the time-n pressure.
    static auto AxisLfU1(const Fields& f, const Grid& g,
                              int l, int m, double dt, double dz, double gamma) -> double;
    static auto AxisLfU2(const Fields& f, const Grid& g,
                              int l, int m, double dt, double dz, double gamma) -> double;
    static auto AxisLfU5(const Fields& f, const Grid& g,
                              int l, int m, double dt, double dz, double gamma) -> double;
    static auto AxisLfU7(const Fields& f, const Grid& g,
                              int l, int m, double dt, double dz, double gamma) -> double;
};
