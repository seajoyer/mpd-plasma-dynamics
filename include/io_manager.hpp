#pragma once

#include <string>
#include "array2d.hpp"
#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Handles all file I/O:
///   - Per-step VTK structured-grid frames (gathered from all MPI ranks)
///   - Run-scoped timestamped output directory
///
/// On construction (rank 0) the directory
///   <cfg.output_dir>/<cfg.run_name>_<DD-MM-YYYY_HH:MM:SS:mmm>/
/// is created.  The path is broadcast so every rank knows it.
///
/// write_frame() is a collective call: all MPI ranks must invoke it
/// together because it internally gathers distributed field data to rank 0
/// before writing the VTK file.
class IOManager {
public:
    IOManager(const SimConfig& cfg, const MPIManager& mpi);

    /// Gather fields from all ranks and write a VTK frame.
    /// Filename: <run_dir>/step_<step04d>.vtk
    /// Must be called collectively by every rank.
    void write_frame(int step, const Fields& f, const Grid& grid);

    /// Convenience: returns the run directory path (same on all ranks).
    const std::string& run_dir() const { return run_dir_; }

private:
    const SimConfig&  cfg_;
    const MPIManager& mpi_;

    std::string run_dir_;   ///< full path to the per-run output directory

    // Global arrays allocated on rank 0 after the first gather.
    Array2D rho_g_, v_z_g_, v_r_g_, v_phi_g_, e_g_;
    Array2D H_z_g_, H_r_g_, H_phi_g_, r_g_;

    // ---- internal helpers ----

    /// Gather one distributed 2-D field to rank-0 global array.
    /// Both src (local) and dst (global, rank-0 only) use [l][m] indexing.
    void gather_global(const Fields& f, const Grid& grid);

    /// Build and write a VTK structured-grid file from the global arrays.
    /// Called by rank 0 only after gather_global().
    void write_vtk(const std::string& filepath) const;
};
