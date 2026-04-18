#pragma once

#include <memory>
#include <string>

#include "iinitial_condition.hpp"

/// VTK-file restart initial condition.
///
/// Reads a structured-grid VTK file that was written by IOManager::WriteFrame
/// and maps every physical field onto the current computational grid.
///
/// ═══════════════════════════════════════════════════════════════════
/// YAML form
/// ═══════════════════════════════════════════════════════════════════
///
///   initial_conditions:
///     vtk_file: "output/run_01-01-2025_12:00:00:000/step_5000.vtk"
///
/// When vtk_file is present any expression fields also listed under
/// initial_conditions are silently ignored; only the VTK file is used.
///
/// ═══════════════════════════════════════════════════════════════════
/// Expected VTK point-data arrays
/// ═══════════════════════════════════════════════════════════════════
///
///   Scalar  "Rho"            → rho
///   Scalar  "Energy"         → e
///   Vector  "Velocity"       → (v_z, v_r, v_phi)  components (0, 1, 2)
///   Vector  "MagneticField"  → (H_z, H_r, H_phi)  components (0, 1, 2)
///
/// Derived fields (p, P) are recomputed from (rho, e, H_*) after all
/// primaries have been assigned — consistent with ExpressionIC.
///
/// ═══════════════════════════════════════════════════════════════════
/// Grid-size compatibility
/// ═══════════════════════════════════════════════════════════════════
///
/// The VTK file carries an (ni × nj) point grid where
///   ni = L_max_old + 1,   nj = M_max_old + 1.
///
/// When the current run uses the same L_max / M_max the mapping is an
/// exact integer lookup (no floating-point blending).  When the grids
/// differ, values are interpolated bilinearly in fractional index space:
///
///   i_frac = l_global_new × L_max_old / L_max_new
///   j_frac = m_global_new × M_max_old / M_max_new
///
/// This preserves the physical (z, fractional-radial) position of each
/// cell independent of resolution, so the same file can seed a refined
/// or coarsened run without loss of physics features.
///
/// ═══════════════════════════════════════════════════════════════════
/// MPI note
/// ═══════════════════════════════════════════════════════════════════
///
/// Every MPI rank opens and reads the file independently.  The file
/// must therefore be accessible from all compute nodes (shared/NFS
/// filesystem).  Memory overhead per rank is
///   (ni × nj × 8 fields × 8 bytes) ≈ 20 MB for an 800 × 400 grid.
class VtkIC : public IInitialCondition {
public:
    /// @param path  Path to the VTK structured-grid file to read.
    ///              Throws std::runtime_error if the file cannot be opened
    ///              or if a required point-data array is missing.
    explicit VtkIC(const std::string& path);

    /// Defined in .cpp where Impl is a complete type.
    ~VtkIC() override;

    void Apply(Fields& fields, const Grid& grid,
               const SimConfig& cfg, int l_start) const override;

    [[nodiscard]] auto Name() const -> std::string override { return "vtk_restart"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
