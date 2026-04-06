#include "config.hpp"

#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <sstream>
#include <stdexcept>

// ============================================================
// Private helpers
// ============================================================

void SimConfig::init() {
    dz = 1.0 / L_max_global;
    dy = 1.0 / M_max;
}

// Serialise a YAML::Node back to a compact string so that it can be stored
// in BCSegmentConfig::params_yaml and re-parsed at factory time.
static std::string node_to_string(const YAML::Node& n) {
    if (!n || n.IsNull()) return {};
    std::ostringstream oss;
    oss << n;
    return oss.str();
}

// Parse one face's boundary-condition list from a YAML sequence node.
//
// Two formats are accepted for maximum convenience:
//
//   # Short form — single type, full face, no range
//   l_lo: inflow
//
//   # Long form — sequence of segment maps
//   m_lo:
//     - range: [0, 320]
//       type: solid_wall
//     - range: [321, -1]
//       type: axis_symmetry
//       params:
//         some_key: some_value
//
// The `range` key is optional; omitting it is equivalent to [-1, -1]
// (full face).
static BCFaceConfig parse_face(const YAML::Node& node) {
    BCFaceConfig face;
    if (!node) return face;

    // Short form: "l_lo: inflow"
    if (node.IsScalar()) {
        BCSegmentConfig seg;
        seg.type = node.as<std::string>();
        face.segments.push_back(std::move(seg));
        return face;
    }

    // Long form: sequence of segment maps.
    if (!node.IsSequence())
        throw std::runtime_error("config: boundary condition face must be a "
                                 "scalar type name or a YAML sequence of segments");

    for (const YAML::Node& item : node) {
        BCSegmentConfig seg;

        if (!item["type"])
            throw std::runtime_error("config: each BC segment must have a 'type' key");
        seg.type = item["type"].as<std::string>();

        if (item["range"]) {
            const YAML::Node& rng = item["range"];
            if (!rng.IsSequence() || rng.size() != 2)
                throw std::runtime_error(
                    "config: BC segment 'range' must be a two-element sequence "
                    "[global_lo, global_hi]  (use -1 as sentinel for face start/end)");
            seg.global_lo = rng[0].as<int>();
            seg.global_hi = rng[1].as<int>();
        }
        // else: both stay at -1 (full-face sentinel)

        if (item["params"])
            seg.params_yaml = node_to_string(item["params"]);

        face.segments.push_back(std::move(seg));
    }

    return face;
}

// ============================================================
// Public loader
// ============================================================

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
        if (n["enabled"])       adaptive_dt       = n["enabled"]      .as<bool>();
        if (n["cfl_number"])    cfl_number         = n["cfl_number"]   .as<double>();
        if (n["growth_factor"]) dt_growth_factor   = n["growth_factor"].as<double>();
        if (n["dt_min"])        dt_min             = n["dt_min"]       .as<double>();
        if (n["dt_max"])        dt_max             = n["dt_max"]       .as<double>();
    }

    // ---- grid --------------------------------------------------------------
    if (auto n = cfg["grid"]) {
        if (n["L_max_global"]) L_max_global = n["L_max_global"].as<int>();
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

    // ---- geometry ----------------------------------------------------------
    if (auto n = cfg["geometry"]) {
        if (n["type"])   geometry.type        = n["type"].as<std::string>();
        if (n["params"]) geometry.params_yaml = node_to_string(n["params"]);
    }

    // ---- boundary conditions -----------------------------------------------
    if (auto n = cfg["boundary_conditions"]) {
        if (n["l_lo"]) bc_l_lo = parse_face(n["l_lo"]);
        if (n["l_hi"]) bc_l_hi = parse_face(n["l_hi"]);
        if (n["m_lo"]) bc_m_lo = parse_face(n["m_lo"]);
        if (n["m_hi"]) bc_m_hi = parse_face(n["m_hi"]);
    }

    // ---- derived quantities ------------------------------------------------
    // Must be recomputed after grid params are loaded.
    init();
}
