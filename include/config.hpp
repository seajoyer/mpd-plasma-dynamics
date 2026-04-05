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
