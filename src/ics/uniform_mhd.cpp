#include "ics/uniform_mhd.hpp"

#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"

// ============================================================
// FieldIC YAML parser
// ============================================================

static auto ParseFieldIC(const YAML::Node& n, const FieldIC& def) -> FieldIC {
    if (!n || n.IsNull()) return def;

    // ---- Scalar shorthand: type name only ----
    if (n.IsScalar()) {
        const auto s = n.as<std::string>();
        if (s == "from_physics") {
            return {.type = FieldICType::FromPhysics};
        }
        if (s == "wall_tangent") {
            return {.type = FieldICType::WallTangent};
        }
        if (s == "linear_free_vortex") {
            return {.type = FieldICType::LinearFreeVortex, .amplitude = 1.0, .factor = 0.9};
        }
        throw std::runtime_error(
            "initial_conditions: unknown FieldIC type '" + s + "'.\n"
            "  Valid scalar types: from_physics, wall_tangent, linear_free_vortex.\n"
            "  For a constant value use: { uniform: <double> }");
    }

    // ---- Map form ----
    if (n.IsMap()) {
        if (n["uniform"]) {
            return {.type = FieldICType::Uniform, .value = n["uniform"].as<double>()};
        }
        if (n["free_vortex"]) {
            const YAML::Node& fv = n["free_vortex"];
            double amp = 1.0;
            if (fv.IsScalar()) {
                amp = fv.as<double>();
            } else if (fv.IsMap() && fv["amplitude"]) {
                amp = fv["amplitude"].as<double>();
            }
            return {.type = FieldICType::FreeVortex, .amplitude = amp};
        }
        if (n["linear_free_vortex"]) {
            const YAML::Node& lfv = n["linear_free_vortex"];
            double amp = 1.0;
            double fac = 0.9;
            if (lfv.IsMap()) {
                if (lfv["amplitude"]) amp = lfv["amplitude"].as<double>();
                if (lfv["factor"])    fac = lfv["factor"].as<double>();
            }
            return {.type = FieldICType::LinearFreeVortex, .amplitude = amp, .factor = fac};
        }
    }

    throw std::runtime_error(
        "initial_conditions: field IC must be a scalar type name or a map "
        "such as { uniform: <v> }, { free_vortex: <a> }, or "
        "{ linear_free_vortex: { amplitude: <a>, factor: <f> } }");
}

// ============================================================
// Constructor
// ============================================================

UniformMhdIC::UniformMhdIC(const YAML::Node& params) {
    // Defaults reproduce the previous hard-coded behaviour of InitPhysical.
    rho_  = ParseFieldIC(params["rho"],
                         {.type = FieldICType::Uniform, .value = 1.0});
    v_z_  = ParseFieldIC(params["v_z"],
                         {.type = FieldICType::Uniform, .value = 0.0});
    v_r_  = ParseFieldIC(params["v_r"],
                         {.type = FieldICType::Uniform, .value = 0.0});
    v_phi_= ParseFieldIC(params["v_phi"],
                         {.type = FieldICType::Uniform, .value = 0.0});
    e_    = ParseFieldIC(params["e"],
                         {.type = FieldICType::FromPhysics});
    H_z_  = ParseFieldIC(params["H_z"],
                         {.type = FieldICType::FromPhysics});
    H_r_  = ParseFieldIC(params["H_r"],
                         {.type = FieldICType::WallTangent});
    H_phi_= ParseFieldIC(params["H_phi"],
                         {.type  = FieldICType::LinearFreeVortex,
                          .amplitude = 1.0,
                          .factor    = 0.9});

    // ---- Validation --------------------------------------------------------

    // FromPhysics is only meaningful for H_z and e.
    auto check_from_physics = [](const FieldIC& ic, const char* field) -> void {
        if (ic.type == FieldICType::FromPhysics) {
            const std::string name{field};
            if (name != "H_z" && name != "e") {
                throw std::runtime_error(
                    "initial_conditions: from_physics is only valid for H_z and e "
                    "(field '" + name + "' does not have a physics-derived default)");
            }
        }
    };
    check_from_physics(rho_,   "rho");
    check_from_physics(v_z_,   "v_z");
    check_from_physics(v_r_,   "v_r");
    check_from_physics(v_phi_, "v_phi");
    check_from_physics(H_r_,   "H_r");
    check_from_physics(H_phi_, "H_phi");

    // WallTangent is only valid for H_r.
    auto check_wall_tangent = [](const FieldIC& ic, const char* field) -> void {
        if (ic.type == FieldICType::WallTangent && std::string{field} != "H_r") {
            throw std::runtime_error(
                std::string("initial_conditions: wall_tangent is only valid for H_r "
                            "(used on '") + field + "')");
        }
    };
    check_wall_tangent(rho_,   "rho");
    check_wall_tangent(v_z_,   "v_z");
    check_wall_tangent(v_r_,   "v_r");
    check_wall_tangent(v_phi_, "v_phi");
    check_wall_tangent(e_,     "e");
    check_wall_tangent(H_z_,   "H_z");
    check_wall_tangent(H_phi_, "H_phi");

    // FreeVortex / LinearFreeVortex are only valid for H_phi.
    auto check_vortex = [](const FieldIC& ic, const char* field) -> void {
        if ((ic.type == FieldICType::FreeVortex ||
             ic.type == FieldICType::LinearFreeVortex) &&
            std::string{field} != "H_phi") {
            throw std::runtime_error(
                std::string("initial_conditions: free_vortex / linear_free_vortex "
                            "is only valid for H_phi (used on '") + field + "')");
        }
    };
    check_vortex(rho_,  "rho");
    check_vortex(v_z_,  "v_z");
    check_vortex(v_r_,  "v_r");
    check_vortex(v_phi_,"v_phi");
    check_vortex(e_,    "e");
    check_vortex(H_z_,  "H_z");
    check_vortex(H_r_,  "H_r");
}

// ============================================================
// Apply
// ============================================================

void UniformMhdIC::Apply(Fields& f, const Grid& grid,
                          const SimConfig& cfg, int l_start) const {
    const double gamma   = cfg.gamma;
    const double beta    = cfg.beta;
    const double H_z0    = cfg.H_z0;
    const double dz      = cfg.dz;

    // Pre-compute from_physics values.
    const double e_physics = beta / (2.0 * (gamma - 1.0));

    // Interior cells only: [1..local_L][1..local_M].
    // Ghost cells are left at zero and will be filled by the first ghost
    // exchange before any stencil computation.
    #pragma omp parallel for collapse(2)
    for (int l = 1; l < f.rows - 1; ++l) {
        for (int m = 1; m < f.cols - 1; ++m) {
            const int    l_global = l_start + l - 1;
            const double z        = l_global * dz;

            // ---- rho ----
            f.rho[l][m] = (rho_.type == FieldICType::Uniform) ? rho_.value : 1.0;

            // ---- velocities ----
            f.v_z  [l][m] = (v_z_  .type == FieldICType::Uniform) ? v_z_.value   : 0.0;
            f.v_r  [l][m] = (v_r_  .type == FieldICType::Uniform) ? v_r_.value   : 0.0;
            f.v_phi[l][m] = (v_phi_.type == FieldICType::Uniform) ? v_phi_.value : 0.0;

            // ---- H_z ----
            f.H_z[l][m] = (H_z_.type == FieldICType::FromPhysics) ? H_z0
                        : (H_z_.type == FieldICType::Uniform)      ? H_z_.value
                        :                                             H_z0;

            // ---- H_r — WallTangent uses the already-set H_z ----
            if (H_r_.type == FieldICType::WallTangent) {
                f.H_r[l][m] = f.H_z[l][m] * grid.r_z[l][m];
            } else {
                f.H_r[l][m] = (H_r_.type == FieldICType::Uniform) ? H_r_.value : 0.0;
            }

            // ---- H_phi ----
            switch (H_phi_.type) {
                case FieldICType::FreeVortex:
                    f.H_phi[l][m] = H_phi_.amplitude * grid.r_0 / grid.r[l][m];
                    break;
                case FieldICType::LinearFreeVortex:
                    f.H_phi[l][m] = H_phi_.amplitude
                                  * (1.0 - H_phi_.factor * z)
                                  * grid.r_0 / grid.r[l][m];
                    break;
                case FieldICType::Uniform:
                    f.H_phi[l][m] = H_phi_.value;
                    break;
                default:
                    f.H_phi[l][m] = grid.r_0 / grid.r[l][m];   // safe fallback
                    break;
            }

            // ---- e ----
            f.e[l][m] = (e_.type == FieldICType::FromPhysics) ? e_physics
                      : (e_.type == FieldICType::Uniform)      ? e_.value
                      :                                           e_physics;

            // ---- Derived scalars — must follow primaries ----
            f.p[l][m] = (gamma - 1.0) * f.rho[l][m] * f.e[l][m];
            f.P[l][m] = f.p[l][m]
                      + 0.5 * (f.H_z  [l][m] * f.H_z  [l][m]
                               + f.H_r  [l][m] * f.H_r  [l][m]
                               + f.H_phi[l][m] * f.H_phi[l][m]);
        }
    }
}
