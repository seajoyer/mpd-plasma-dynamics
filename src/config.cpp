#include "config.hpp"

#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <stdexcept>

// ---- private ---------------------------------------------------------------

void SimConfig::init() {
    dz = 1.0 / L_max_global;
    dy = 1.0 / M_max;
}

// ---- public ----------------------------------------------------------------

void SimConfig::load(const std::string& path) {
    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load config file '") + path + "': " + e.what());
    }

    // ---- physics -----------------------------------------------------------
    if (auto n = cfg["physics"]) {
        if (n["gamma"]) gamma = n["gamma"].as<double>();
        if (n["beta"])  beta  = n["beta"] .as<double>();
        if (n["H_z0"])  H_z0  = n["H_z0"] .as<double>();
    }

    // ---- time --------------------------------------------------------------
    if (auto n = cfg["time"]) {
        if (n["T"])  T  = n["T"] .as<double>();
        if (n["dt"]) dt = n["dt"].as<double>();
    }

    // ---- adaptive time step ------------------------------------------------
    if (auto n = cfg["adaptive_dt"]) {
        if (n["enabled"])      adaptive_dt      = n["enabled"]     .as<bool>();
        if (n["cfl_number"])   cfl_number        = n["cfl_number"]  .as<double>();
        if (n["growth_factor"])dt_growth_factor  = n["growth_factor"].as<double>();
        if (n["dt_min"])       dt_min            = n["dt_min"]      .as<double>();
        if (n["dt_max"])       dt_max            = n["dt_max"]      .as<double>();
    }

    // ---- grid --------------------------------------------------------------
    if (auto n = cfg["grid"]) {
        if (n["L_max_global"]) L_max_global = n["L_max_global"].as<int>();
        if (n["L_end"])        L_end        = n["L_end"]       .as<int>();
        if (n["M_max"])        M_max        = n["M_max"]       .as<int>();
    }

    // ---- convergence -------------------------------------------------------
    if (auto n = cfg["convergence"]) {
        if (n["threshold"])       convergence_threshold = n["threshold"]      .as<double>();
        if (n["check_frequency"]) check_frequency       = n["check_frequency"].as<int>();
    }

    // ---- output ------------------------------------------------------------
    if (auto n = cfg["output"]) {
        if (n["directory"]) output_dir = n["directory"].as<std::string>();
        if (n["run_name"])  run_name   = n["run_name"] .as<std::string>();
        if (n["vtk_step"])  vtk_step   = n["vtk_step"] .as<int>();
    }

    // ---- parallelism -------------------------------------------------------
    if (auto n = cfg["parallel"]) {
        if (n["openmp_threads"]) openmp_threads = n["openmp_threads"].as<int>();
        if (n["mpi_dims_l"])     mpi_dims_l     = n["mpi_dims_l"]    .as<int>();
        if (n["mpi_dims_m"])     mpi_dims_m     = n["mpi_dims_m"]    .as<int>();
    }

    // Derived quantities must be recomputed after grid params are loaded.
    init();
}
