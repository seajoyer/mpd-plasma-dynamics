#include "config.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <sstream>
#include <stdexcept>

// ============================================================
// Private helpers
// ============================================================

using std::move;

void SimConfig::Init() {
    dz = 1.0 / L_max_global;
    dy = 1.0 / M_max;
}

// ---- FieldCond parser -------------------------------------------------------
//
// Accepts three YAML forms for a single field condition:
//
//   neumann                          # short: type name as scalar string
//   wall_tangent
//   axis_lf
//   hphi_r0_over_r
//   { dirichlet: 1.0 }               # map with implicit value key
//   { type: dirichlet, value: 1.0 }  # verbose map
//
// Unrecognised keys throw std::runtime_error.

static auto ParseFieldCond(const YAML::Node& n) -> FieldCond {
    if (!n || n.IsNull()) {
        return {};  // default: Neumann
    }

    // ---- Short scalar form: type name only ----
    if (n.IsScalar()) {
        const auto s = n.as<std::string>();
        if (s == "neumann") return {.type = FieldCondType::Neumann};
        if (s == "wall_tangent") return {.type = FieldCondType::WallTangent};
        if (s == "axis_lf") return {.type = FieldCondType::AxisLF};
        if (s == "hphi_r0_over_r") return {.type = FieldCondType::HPhi_r0_over_r};
        throw std::runtime_error(
            "config: unknown field condition '" + s +
            "'.\n"
            "  Valid scalar types: neumann, wall_tangent, axis_lf, hphi_r0_over_r.\n"
            "  For a fixed value use: { dirichlet: <value> }");
    }

    // ---- Map form ----
    if (n.IsMap()) {
        // Short map: { dirichlet: 1.0 }
        if (n["dirichlet"]) {
            return {.type = FieldCondType::Dirichlet,
                    .value = n["dirichlet"].as<double>()};
        }

        // Verbose map: { type: ..., value: ... }
        if (n["type"]) {
            const auto t = n["type"].as<std::string>();
            if (t == "neumann") return {.type = FieldCondType::Neumann};
            if (t == "wall_tangent") return {.type = FieldCondType::WallTangent};
            if (t == "axis_lf") return {.type = FieldCondType::AxisLF};
            if (t == "hphi_r0_over_r") return {.type = FieldCondType::HPhi_r0_over_r};
            if (t == "dirichlet") {
                if (!n["value"]) {
                    throw std::runtime_error(
                        "config: dirichlet condition requires a 'value' key");
                }
                return {.type = FieldCondType::Dirichlet,
                        .value = n["value"].as<double>()};
            }
            throw std::runtime_error("config: unknown field condition type '" + t + "'");
        }
    }

    throw std::runtime_error(
        "config: field condition must be a scalar type name or a map "
        "{ dirichlet: <value> } / { type: ..., value: ... }");
}

// ---- Face parser ------------------------------------------------------------
//
// Each face is a YAML sequence of segment maps.  Each map may contain:
//
//   range     (optional)  [global_lo, global_hi]  — negative = sentinel
//   rho / v_z / v_r / v_phi / e / H_z / H_r / H_phi
//             (all optional, default Neumann)      — FieldCond specification

static auto ParseFace(const YAML::Node& node) -> BCFaceConfig {
    BCFaceConfig face;
    if (!node) return face;

    if (!node.IsSequence()) {
        throw std::runtime_error(
            "config: each boundary-condition face must be a YAML sequence "
            "of segment maps");
    }

    for (const YAML::Node& item : node) {
        if (!item.IsMap()) {
            throw std::runtime_error("config: each BC segment must be a YAML map");
        }

        BCSegmentConfig seg;

        // Optional range
        if (item["range"]) {
            const YAML::Node& rng = item["range"];
            if (!rng.IsSequence() || rng.size() != 2) {
                throw std::runtime_error(
                    "config: BC segment 'range' must be a two-element sequence "
                    "[global_lo, global_hi]  (use -1 as sentinel for face start/end)");
            }
            seg.global_lo = rng[0].as<int>();
            seg.global_hi = rng[1].as<int>();
        }

        // Per-field conditions (all optional; absent = Neumann)
        seg.rho = ParseFieldCond(item["rho"]);
        seg.v_z = ParseFieldCond(item["v_z"]);
        seg.v_r = ParseFieldCond(item["v_r"]);
        seg.v_phi = ParseFieldCond(item["v_phi"]);
        seg.e = ParseFieldCond(item["e"]);
        seg.H_z = ParseFieldCond(item["H_z"]);
        seg.H_r = ParseFieldCond(item["H_r"]);
        seg.H_phi = ParseFieldCond(item["H_phi"]);

        face.segments.push_back(std::move(seg));
    }

    return face;
}

// ============================================================
// Public loader
// ============================================================

void SimConfig::Load(const std::string& path) {
    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(std::string("Failed to load config file '") + path +
                                 "': " + e.what());
    }

    // ---- physics -----------------------------------------------------------
    if (auto n = cfg["physics"]) {
        if (n["gamma"]) gamma = n["gamma"].as<double>();
        if (n["beta"]) beta = n["beta"].as<double>();
        if (n["H_z0"]) H_z0 = n["H_z0"].as<double>();
    }

    // ---- time --------------------------------------------------------------
    if (auto n = cfg["time"]) {
        if (n["T"]) T = n["T"].as<double>();
        if (n["dt"]) dt = n["dt"].as<double>();
    }

    // ---- adaptive time step ------------------------------------------------
    if (auto n = cfg["adaptive_dt"]) {
        if (n["enabled"]) adaptive_dt = n["enabled"].as<bool>();
        if (n["cfl_number"]) cfl_number = n["cfl_number"].as<double>();
        if (n["growth_factor"]) dt_growth_factor = n["growth_factor"].as<double>();
        if (n["dt_min"]) dt_min = n["dt_min"].as<double>();
        if (n["dt_max"]) dt_max = n["dt_max"].as<double>();
    }

    // ---- grid --------------------------------------------------------------
    if (auto n = cfg["grid"]) {
        if (n["L_max_global"]) L_max_global = n["L_max_global"].as<int>();
        if (n["M_max"]) M_max = n["M_max"].as<int>();
    }

    // ---- convergence -------------------------------------------------------
    if (auto n = cfg["convergence"]) {
        if (n["threshold"]) convergence_threshold = n["threshold"].as<double>();
        if (n["check_frequency"]) check_frequency = n["check_frequency"].as<int>();
    }

    // ---- output ------------------------------------------------------------
    if (auto n = cfg["output"]) {
        if (n["directory"]) output_dir = n["directory"].as<std::string>();
        if (n["run_name"]) run_name = n["run_name"].as<std::string>();
        if (n["vtk_step"]) vtk_step = n["vtk_step"].as<int>();
    }

    // ---- parallelism -------------------------------------------------------
    if (auto n = cfg["parallel"]) {
        if (n["openmp_threads"]) openmp_threads = n["openmp_threads"].as<int>();
        if (n["mpi_dims_l"]) mpi_dims_l = n["mpi_dims_l"].as<int>();
        if (n["mpi_dims_m"]) mpi_dims_m = n["mpi_dims_m"].as<int>();
    }

    // ---- geometry ----------------------------------------------------------
    if (auto n = cfg["geometry"]) {
        if (n["type"]) geometry.type = n["type"].as<std::string>();
        if (n["params"]) {
            std::ostringstream oss;
            oss << n["params"];
            geometry.params_yaml = oss.str();
        }
    }

    // ---- initial conditions ------------------------------------------------
    if (auto n = cfg["initial_conditions"]) {
        if (n["type"]) initial_conditions.type = n["type"].as<std::string>();
        if (n["params"]) {
            std::ostringstream oss;
            oss << n["params"];
            initial_conditions.params_yaml = oss.str();
        }
    }

    // ---- boundary conditions -----------------------------------------------
    if (auto n = cfg["boundary_conditions"]) {
        if (n["l_lo"]) bc_l_lo = ParseFace(n["l_lo"]);
        if (n["l_hi"]) bc_l_hi = ParseFace(n["l_hi"]);
        if (n["m_lo"]) bc_m_lo = ParseFace(n["m_lo"]);
        if (n["m_hi"]) bc_m_hi = ParseFace(n["m_hi"]);
    }

    // ---- derived quantities ------------------------------------------------
    // Must be recomputed after grid params are loaded.
    Init();
}
