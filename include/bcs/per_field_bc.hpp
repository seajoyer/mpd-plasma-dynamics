#pragma once

#include "iboundary_condition.hpp"
#include "config.hpp"
#include "face_bc.hpp"     // for FaceBC::Face

/// Generic boundary condition that applies an independent FieldCond to each
/// of the eight physical variables (rho, v_z, v_r, v_phi, e, H_z, H_r, H_phi)
/// at a boundary segment.
///
/// Supported condition types (see FieldCondType for full documentation):
///
///   Neumann         Zero-gradient: copy from the first interior cell.
///   Dirichlet       Fix to a constant value.
///   WallTangent     Radial component derived from wall-slope metric.
///                   v_r = v_z · r_z   or   H_r = H_z · r_z.
///                   Valid only for v_r / H_r on M_LO and M_HI faces.
///   AxisLF          Half-stencil Lax–Friedrichs update for the axis of symmetry.
///                   Valid only for rho / v_z / e / H_z on the M_LO face.
///                   Writes the corresponding conservative u_* array directly;
///                   physical reconstruction is deferred to the end-of-step
///                   Fields::update_physical_from_u() call in Solver::advance().
///   HPhi_r0_over_r  Free-vortex inlet profile for H_phi: H_phi = r_0 / r.
///                   Enforces H_phi · r = r_0 = const (no azimuthal current).
///                   Valid only for H_phi.
///
/// Step flow for each boundary cell
/// ─────────────────────────────────
///   1. Apply Neumann / Dirichlet / WallTangent / HPhi_r0_over_r to physical vars.
///   2. Call rebuild_u_from_physical() to keep conservative arrays consistent.
///   3. Overwrite u_* for any AxisLF fields with the one-sided LF stencil.
///
/// Step 3 is a no-op when no AxisLF conditions are present.
class PerFieldBC : public IBoundaryCondition {
public:
    /// @param face  Which of the four Cartesian faces this segment belongs to.
    ///              Required to determine the fixed index and Neumann direction.
    /// @param seg   Per-field condition configuration read from config.yaml.
    PerFieldBC(enum FaceBC::Face face, const BCSegmentConfig& seg);

    void        Apply(BCContext& ctx) const override;
    [[nodiscard]] auto Name()                const -> std::string override { return "per_field"; }

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

    bool has_axis_lf_; ///< true if any field uses AxisLF — avoids a branch-per-cell scan

    // ---- AxisLF stencil helpers (M_LO face, m = 1) ----
    // Each method mirrors the corresponding update in the original
    // axis_symmetry_bc.cpp, expressed as a free-standing formula so that
    // the caller can skip any field that is not configured as AxisLF.
    static auto AxisLfU1(const Fields& f, const Grid& g,
                              int l, int m, double dt, double dz) -> double;
    static auto AxisLfU2(const Fields& f, const Grid& g,
                              int l, int m, double dt, double dz) -> double;
    static auto AxisLfU5(const Fields& f, const Grid& g,
                              int l, int m, double dt, double dz) -> double;
    static auto AxisLfU7(const Fields& f, const Grid& g,
                              int l, int m, double dt, double dz) -> double;
};
