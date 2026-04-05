#pragma once

#include <string>

/// All tuneable parameters for the MHD simulation.
///
/// Primary source is config.yaml (loaded via SimConfig::load()).
/// Derived quantities (dz, dy) are computed automatically after loading.
struct SimConfig {
    // ---- physics ----
    double gamma = 1.67;
    double beta  = 0.05;
    double H_z0  = 0.25;

    // ---- time integration ----
    double T  = 0.5;
    double dt = 0.000025;

    // ---- global grid ----
    int L_max_global = 800;
    int L_end        = 320;   ///< axial index where inner-wall BC switches regime
    int M_max        = 400;

    // ---- convergence ----
    double convergence_threshold = 0.0;   ///< 0 → disabled
    int    check_frequency       = 100;

    // ---- output ----
    std::string output_dir = "output";  ///< top-level directory for all runs
    std::string run_name   = "run";     ///< prefix of the per-run sub-directory
    int         vtk_step   = 100;       ///< write VTK frame every N steps (0 → final only)

    // ---- parallelism ----
    int openmp_threads = 0;   ///< 0 → defer to OMP_NUM_THREADS env var

    /// MPI Cartesian decomposition hints.
    /// 0 means "let MPI_Dims_create decide".
    ///
    /// For pure-MPI runs (openmp_threads == 0 or 1) it is almost always
    /// faster to set mpi_dims_m = 1 (1-D decomposition along L only) because:
    ///   - the inner m-loop stays at M_max iterations → auto-vectorised;
    ///   - MPI ghost exchange never needs to pack non-contiguous columns;
    ///   - MPI message count is halved (only 2 neighbours instead of 4).
    ///
    /// For hybrid MPI+OpenMP runs set mpi_dims_m > 1 and let OpenMP threads
    /// cover the M dimension within each rank.
    int mpi_dims_l = 0;   ///< 0 = auto
    int mpi_dims_m = 0;   ///< 0 = auto; set to 1 for pure-MPI runs

    // ---- derived (computed by load / init) ----
    double dz{};
    double dy{};

    /// Ensure dz and dy are consistent with the default grid parameters.
    SimConfig() { init(); }

    /// Load all parameters from a YAML file and compute derived quantities.
    /// @param path  path to the YAML config file (default: "config.yaml")
    void load(const std::string& path = "config.yaml");

private:
    /// Compute dz and dy from the primary grid parameters.
    void init();
};
