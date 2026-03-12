#pragma once

/// All compile-time tuneable parameters for the MHD simulation.
/// Derived quantities (dz, dy) are computed by init().
/// Command-line arguments are parsed by parse_args().
struct SimConfig {
    // ---- physics ----
    double gamma = 1.67;
    double beta  = 0.05;
    double H_z0  = 0.25;

    // ---- convergence ----
    double convergence_threshold = 0.0;   ///< 0 → disabled
    int    check_frequency       = 100;
    int    animate = 0;

    // ---- time integration ----
    double T  = 0.5;
    double dt = 0.000025;

    // ---- global grid ----
    int L_max_global = 800;
    int L_end        = 320;   ///< axial index where inner BC switches regime
    int M_max        = 400;

    // ---- derived (set by init()) ----
    double dz{};
    double dy{};

    /// Compute derived quantities from the primary parameters.
    void init();

    /// Parse --converge <value> from argv (skips argv[0] and argv[1]).
    void parse_args(int argc, char* argv[], int rank);
};
