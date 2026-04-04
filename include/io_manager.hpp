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
/// together.  Point-to-point gather strategy:
///   - Every non-zero rank sends (local_L × local_M) data blocks for each
///     field, along with its (l_start, m_start, local_L, local_M) envelope,
///     to rank 0.
///   - Rank 0 receives sequentially and places each block into the matching
///     region of the global array.
///   - Rank 0 then writes the VTK file.
class IOManager {
public:
    IOManager(const SimConfig& cfg, const MPIManager& mpi);

    /// Gather fields from all ranks and write a VTK frame.
    /// Filename: <run_dir>/step_<step04d>.vtk
    /// Must be called collectively by every rank.
    void write_frame(int step, const Fields& f, const Grid& grid);

    /// Returns the run directory path (same on all ranks after construction).
    const std::string& run_dir() const { return run_dir_; }

private:
    const SimConfig&  cfg_;
    const MPIManager& mpi_;
    std::string run_dir_;

    // Global arrays, allocated on rank 0 the first time gather_global runs.
    Array2D rho_g_, v_z_g_, v_r_g_, v_phi_g_, e_g_;
    Array2D H_z_g_, H_r_g_, H_phi_g_, r_g_;
    Array2D rank_g_;   ///< MPI rank that owns each cell — for decomposition visualisation.

    // ---- internal helpers ----

    /// Gather all distributed field arrays to rank-0 global arrays using
    /// point-to-point communication.
    void gather_global(const Fields& f, const Grid& grid);

    /// Flatten a local interior block [1..local_L][1..local_M] into a
    /// contiguous send buffer of length local_L * local_M.
    static void pack_field(const Array2D& src, int local_L, int local_M,
                           std::vector<double>& buf);

    /// Scatter a flat buffer of length block_L * block_M into the global
    /// array starting at (gl, gm).
    static void unpack_into_global(Array2D& dst,
                                   const std::vector<double>& buf,
                                   int gl, int gm,
                                   int block_L, int block_M);

    /// Fill every cell in the global rank array owned by a given MPI rank.
    /// Called on rank 0 only — no communication required because the block
    /// geometry is already known from the gather envelope.
    static void fill_rank_block(Array2D& dst, double rank_id,
                                int gl, int gm,
                                int block_L, int block_M);

    /// Build and write a VTK structured-grid file from the global arrays.
    /// Called by rank 0 only after gather_global().
    void write_vtk(const std::string& filepath) const;
};
