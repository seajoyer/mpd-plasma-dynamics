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
    dz = 1.0 / L_max;
    dy = 1.0 / M_max;
}

// ---- FieldCond parser -------------------------------------------------------
//
// Accepts the following YAML forms for a single field condition:
//
//   Neumann (zero-gradient copy + optional offset):
//     neumann                         →  Neumann, offset = 0
//     { neumann: 0.5 }                →  Neumann, offset = 0.5 (f_bc = f_nb + 0.5)
//
//   Dirichlet (fixed value, defaults to 0):
//     dirichlet                       →  Dirichlet, value = 0
//     { dirichlet: 1.0 }              →  Dirichlet, value = 1.0
//
//   Expression (arbitrary formula compiled via exprtk):
//     "v_z * r_z"                     →  Expression, expr = "v_z * r_z"
//     "r_0 / r"                       →  Expression, expr = "r_0 / r"
//     { expr: "v_z_nb * r_z_nb" }     →  Expression, expr = "v_z_nb * r_z_nb"
//
//   Axis Lax–Friedrichs (special half-stencil, M_LO only):
//     axis_lf                         →  AxisLF
//
//   Verbose map form (all conditions):
//     { type: neumann }
//     { type: neumann, gradient: 0.5 }
//     { type: dirichlet }
//     { type: dirichlet, value: 1.0 }
//     { type: expression, expr: "..." }
//     { type: axis_lf }
//
// Removed presets — helpful errors with migration suggestions:
//   wall_tangent      → "v_z_nb * r_z_nb"  (M_LO)  /  "v_z * r_z"  (M_HI)
//   hphi_r0_over_r    → "r_0 / r"

static auto ParseFieldCond(const YAML::Node& n) -> FieldCond {
    if (!n || n.IsNull()) {
        return {};  // default: Neumann with offset 0 (zero-gradient copy)
    }

    // ---- Scalar form --------------------------------------------------------
    if (n.IsScalar()) {
        const auto s = n.as<std::string>();

        if (s == "neumann")   return {.type = FieldCondType::Neumann};
        if (s == "dirichlet") return {.type = FieldCondType::Dirichlet, .value = 0.0};
        if (s == "axis_lf")   return {.type = FieldCondType::AxisLF};

        // ---- Removed-preset migration errors --------------------------------
        if (s == "wall_tangent") {
            throw std::runtime_error(
                "config: 'wall_tangent' is no longer a built-in preset.\n"
                "  Replace it with an expression that encodes the wall-tangent\n"
                "  condition explicitly:\n"
                "\n"
                "    On M_LO (inner wall) — uses interior-neighbour slope:\n"
                "      v_r: \"v_z_nb * r_z_nb\"\n"
                "      H_r: \"H_z_nb * r_z_nb\"\n"
                "\n"
                "    On M_HI (outer wall) — uses wall-cell slope:\n"
                "      v_r: \"v_z * r_z\"\n"
                "      H_r: \"H_z * r_z\"\n"
                "\n"
                "  Note: 'v_z' and 'H_z' here refer to the wall-cell values\n"
                "  (already evaluated earlier in the same cell's update).\n"
                "  'v_z_nb' / 'H_z_nb' refer to the interior-neighbour cell.");
        }

        if (s == "hphi_r0_over_r") {
            throw std::runtime_error(
                "config: 'hphi_r0_over_r' is no longer a built-in preset.\n"
                "  Replace it with the equivalent expression:\n"
                "\n"
                "    H_phi: \"r_0 / r\"\n"
                "\n"
                "  This enforces H_phi * r = r_0 = const (free-vortex / no-current\n"
                "  azimuthal field profile).");
        }

        // ---- Any other scalar string → expression ---------------------------
        // Anything that doesn't match a keyword is treated as a formula.
        // exprtk will report a clear error at construction time if the string
        // is not a valid mathematical expression.
        return {.type = FieldCondType::Expression, .expr_str = s};
    }

    // ---- Map form -----------------------------------------------------------
    if (n.IsMap()) {

        // { neumann: 0.5 }  — short map with offset value
        if (n["neumann"]) {
            const auto& v = n["neumann"];
            if (!v.IsScalar()) {
                throw std::runtime_error(
                    "config: 'neumann' offset must be a scalar number "
                    "(e.g. { neumann: 0.5 })");
            }
            return {.type = FieldCondType::Neumann, .value = v.as<double>()};
        }

        // { dirichlet: 1.0 }  — short map with fixed value
        if (n["dirichlet"]) {
            const auto& v = n["dirichlet"];
            if (!v.IsScalar()) {
                throw std::runtime_error(
                    "config: 'dirichlet' value must be a scalar number "
                    "(e.g. { dirichlet: 1.0 })");
            }
            return {.type = FieldCondType::Dirichlet, .value = v.as<double>()};
        }

        // { expr: "..." }  — short expression map
        if (n["expr"]) {
            const auto& v = n["expr"];
            if (!v.IsScalar()) {
                throw std::runtime_error(
                    "config: 'expr' value must be a scalar expression string "
                    "(e.g. { expr: \"v_z * r_z\" })");
            }
            return {.type = FieldCondType::Expression,
                    .expr_str = v.as<std::string>()};
        }

        // { type: ..., [value/gradient/expr]: ... }  — verbose map
        if (n["type"]) {
            const auto t = n["type"].as<std::string>();

            if (t == "neumann") {
                double offset = 0.0;
                if (n["gradient"]) offset = n["gradient"].as<double>();
                else if (n["value"]) offset = n["value"].as<double>();
                return {.type = FieldCondType::Neumann, .value = offset};
            }

            if (t == "dirichlet") {
                double val = 0.0;
                if (n["value"]) val = n["value"].as<double>();
                return {.type = FieldCondType::Dirichlet, .value = val};
            }

            if (t == "axis_lf") {
                return {.type = FieldCondType::AxisLF};
            }

            if (t == "expression" || t == "expr") {
                if (!n["expr"]) {
                    throw std::runtime_error(
                        "config: { type: expression } requires an 'expr' key "
                        "containing the formula string\n"
                        "  Example: { type: expression, expr: \"v_z * r_z\" }");
                }
                return {.type     = FieldCondType::Expression,
                        .expr_str = n["expr"].as<std::string>()};
            }

            // ---- Removed preset names in verbose form ----
            if (t == "wall_tangent") {
                throw std::runtime_error(
                    "config: { type: wall_tangent } is no longer supported.\n"
                    "  See the 'wall_tangent' migration note above.");
            }
            if (t == "hphi_r0_over_r") {
                throw std::runtime_error(
                    "config: { type: hphi_r0_over_r } is no longer supported.\n"
                    "  Use: H_phi: \"r_0 / r\"");
            }

            throw std::runtime_error(
                "config: unknown field condition type '" + t + "'.\n"
                "  Valid types: neumann, dirichlet, expression, axis_lf.\n"
                "  For an expression use a quoted string directly:  v_r: \"v_z * r_z\"");
        }

        throw std::runtime_error(
            "config: field condition map must contain one of: 'neumann', "
            "'dirichlet', 'expr', or 'type'");
    }

    throw std::runtime_error(
        "config: field condition must be:\n"
        "  - a keyword scalar:           neumann / dirichlet / axis_lf\n"
        "  - a quoted expression string: \"v_z * r_z\"\n"
        "  - a short map:                { dirichlet: 1.0 } / { neumann: 0.5 } / { expr: \"...\" }\n"
        "  - a verbose map:              { type: dirichlet, value: 1.0 }");
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

        // Per-field conditions (all optional; absent = Neumann, offset = 0)
        seg.rho   = ParseFieldCond(item["rho"]);
        seg.v_z   = ParseFieldCond(item["v_z"]);
        seg.v_r   = ParseFieldCond(item["v_r"]);
        seg.v_phi = ParseFieldCond(item["v_phi"]);
        seg.e     = ParseFieldCond(item["e"]);
        seg.H_z   = ParseFieldCond(item["H_z"]);
        seg.H_r   = ParseFieldCond(item["H_r"]);
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
        if (n["beta"])  beta  = n["beta"].as<double>();
        if (n["H_z0"])  H_z0  = n["H_z0"].as<double>();
    }

    // ---- time --------------------------------------------------------------
    if (auto n = cfg["time"]) {
        if (n["T"])  T  = n["T"].as<double>();
        if (n["dt"]) dt = n["dt"].as<double>();
    }

    // ---- adaptive time step ------------------------------------------------
    if (auto n = cfg["adaptive_dt"]) {
        if (n["enabled"])       adaptive_dt      = n["enabled"].as<bool>();
        if (n["cfl_number"])    cfl_number       = n["cfl_number"].as<double>();
        if (n["growth_factor"]) dt_growth_factor = n["growth_factor"].as<double>();
        if (n["dt_min"])        dt_min           = n["dt_min"].as<double>();
        if (n["dt_max"])        dt_max           = n["dt_max"].as<double>();
    }

    // ---- grid --------------------------------------------------------------
    if (auto n = cfg["grid"]) {
        if (n["L_max"]) L_max = n["L_max"].as<int>();
        if (n["M_max"]) M_max = n["M_max"].as<int>();
    }

    // ---- convergence -------------------------------------------------------
    if (auto n = cfg["convergence"]) {
        if (n["threshold"])       convergence_threshold = n["threshold"].as<double>();
        if (n["check_frequency"]) check_frequency       = n["check_frequency"].as<int>();
    }

    // ---- output ------------------------------------------------------------
    if (auto n = cfg["output"]) {
        if (n["directory"]) output_dir = n["directory"].as<std::string>();
        if (n["run_name"])  run_name   = n["run_name"].as<std::string>();
        if (n["vtk_step"])  vtk_step   = n["vtk_step"].as<int>();
    }

    // ---- parallelism -------------------------------------------------------
    if (auto n = cfg["parallel"]) {
        if (n["openmp_threads"]) openmp_threads = n["openmp_threads"].as<int>();
        if (n["mpi_dims_l"])     mpi_dims_l     = n["mpi_dims_l"].as<int>();
        if (n["mpi_dims_m"])     mpi_dims_m     = n["mpi_dims_m"].as<int>();
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
        if (n["type"]) {
            throw std::runtime_error(
                "config: 'initial_conditions.type' is no longer supported — "
                "ExpressionIC is always used.\n"
                "  Remove the 'type' key.  Field expressions now live directly\n"
                "  under 'initial_conditions:', not under 'initial_conditions.params:'.\n"
                "  Example:\n"
                "    initial_conditions:\n"
                "      rho: 1.0\n"
                "      v_z: 0.1");
        }
        if (n["params"]) {
            throw std::runtime_error(
                "config: 'initial_conditions.params' is no longer supported.\n"
                "  Move field expressions one level up, directly under "
                "'initial_conditions:'.\n"
                "  Example:\n"
                "    initial_conditions:\n"
                "      rho: 1.0\n"
                "      v_z: 0.1");
        }
        std::ostringstream oss;
        oss << n;
        initial_conditions.params_yaml = oss.str();
    }

    // ---- boundary conditions -----------------------------------------------
    if (auto n = cfg["boundary_conditions"]) {
        if (n["l_lo"]) bc_l_lo = ParseFace(n["l_lo"]);
        if (n["l_hi"]) bc_l_hi = ParseFace(n["l_hi"]);
        if (n["m_lo"]) bc_m_lo = ParseFace(n["m_lo"]);
        if (n["m_hi"]) bc_m_hi = ParseFace(n["m_hi"]);
    }

    // ---- derived quantities ------------------------------------------------
    Init();
}
