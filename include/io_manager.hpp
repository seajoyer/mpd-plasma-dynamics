#pragma once

#include <string>
#include <vector>
#include "array2d.hpp"
#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

/// Handles all file I/O:
///   - Per-rank Tecplot animation frames (.plt)
///   - Global VTK structured grid output
///   - Final global Tecplot result file
///
/// The gather step (local → global) is performed by gather_global()
/// and must be called collectively by every rank before the write routines.
class IOManager {
public:
    IOManager(const SimConfig& cfg, const MPIManager& mpi);

    // ---- animation (called every N steps, per-rank) ----

    /// Write one Tecplot animation frame for this rank.
    void write_animate_frame(int step, const Fields& f,
                              const Grid& grid) const;

    // ---- final output (called once at end of simulation) ----

    /// Gather local field data to rank-0 global arrays.
    /// Must be called collectively.  After return, the global arrays on
    /// rank 0 are filled; on other ranks they remain empty.
    void gather_global(const Fields& f, const Grid& grid);

    /// Write VTK structured-grid file (rank 0 only, call after gather).
    void write_vtk(const std::string& filename) const;

    /// Write Tecplot FEPOINT file (rank 0 only, call after gather).
    void write_tecplot(const std::string& filename) const;

private:
    const SimConfig&  cfg_;
    const MPIManager& mpi_;

    // Global arrays (non-null on rank 0 after gather_global)
    Array2D rho_g_, v_z_g_, v_r_g_, v_phi_g_, e_g_;
    Array2D H_z_g_, H_r_g_, H_phi_g_, r_g_;
};
