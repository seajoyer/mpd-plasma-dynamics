#pragma once

#include <string>
#include <vector>
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
/// WriteFrame() is a collective call: all MPI ranks must invoke it
/// together.  Gather strategy:
///
///   - The MPI decomposition is fixed at startup, so per-rank block sizes
///     and global offsets are gathered *once* in the constructor via
///     MPI_Gather (a single 4-int envelope per rank) and cached.
///   - On each WriteFrame() every rank packs its 9 local field blocks
///     into one contiguous send buffer (length = nfields × local_L ×
///     local_M).  A single MPI_Gatherv collects every rank's buffer onto
///     rank 0 in one collective.
///   - Rank 0 unpacks each rank's segment into the 9 global Array2D
///     buffers using the cached envelope.
///   - Rank 0 then writes the VTK file.
class IOManager {
public:
    IOManager(const SimConfig& cfg, const MPIManager& mpi);

    /// Gather fields from all ranks and write a VTK frame.
    /// Filename: <run_dir>/step_<step04d>.vtk
    /// Must be called collectively by every rank.
    void WriteFrame(int step, const Fields& f, const Grid& grid);

    /// Returns the run directory path (same on all ranks after construction).
    [[nodiscard]] auto RunDir() const -> const std::string& { return run_dir_; }

private:
    const SimConfig&  cfg_;
    const MPIManager& mpi_;
    std::string run_dir_;

    // Number of distributed fields gathered per frame:
    //   rho, v_z, v_r, v_phi, e, H_z, H_r, H_phi, r          (= 9)
    static constexpr int kNumFields = 9;

    // ---- Cached gather metadata (built once in the constructor) ---------
    std::vector<int> block_L_;    ///< [rank] = local_L  on that rank
    std::vector<int> block_M_;    ///< [rank] = local_M  on that rank
    std::vector<int> gl_;         ///< [rank] = l_start  on that rank
    std::vector<int> gm_;         ///< [rank] = m_start  on that rank

    // Per-rank length, in doubles, of the field payload in the gathered
    // buffer (= kNumFields * block_L * block_M).  And displacements giving
    // each rank's offset into the gather recv buffer.  Both have size
    // mpi_.size on rank 0; empty elsewhere.
    std::vector<int> recv_counts_;
    std::vector<int> recv_displs_;

    // Reusable buffers — keep capacity across WriteFrame calls.
    std::vector<double> send_buf_;   ///< per-step pack of all 9 fields on this rank
    std::vector<double> recv_buf_;   ///< rank-0 ragged gather of every rank's send

    // Global arrays, allocated on rank 0 the first time gather_global runs.
    Array2D rho_g_, v_z_g_, v_r_g_, v_phi_g_, e_g_;
    Array2D H_z_g_, H_r_g_, H_phi_g_, r_g_;
    Array2D rank_g_;   ///< MPI rank that owns each cell — for decomposition visualisation.

    // ---- internal helpers ----

    /// One-shot: gather per-rank envelopes onto rank 0 and pre-compute
    /// recv_counts_ / recv_displs_.  Called from the constructor.
    void BuildGatherMetadata();

    /// Pack this rank's nine local interior blocks into send_buf_ in field
    /// order: rho, v_z, v_r, v_phi, e, H_z, H_r, H_phi, grid.r.  Each block
    /// is stored row-major in (local_L × local_M) without ghost rows/cols.
    void PackLocalBlocks(const Fields& f, const Grid& grid);

    /// Rank-0 only: unpack the gathered ragged buffer into the nine global
    /// Array2D destinations, and (if requested) stamp the rank field.
    void UnpackGlobalBlocks();

    /// Build and write a VTK structured-grid file from the global arrays.
    /// Called by rank 0 only after the gather.
    void WriteVtk(const std::string& filepath) const;
};
