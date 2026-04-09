#include "ics/expression_ic.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

// exprtk is a header-only expression evaluator.  Including it here (and only
// here) keeps compile times for the rest of the project unaffected.
#define exprtk_disable_string_capabilities   // not needed — saves compile time
#define exprtk_disable_rtl_io_file           // no file I/O needed
#define exprtk_disable_rtl_vecops            // no vector ops needed
#include <exprtk.hpp>

#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"

// ============================================================
// Impl — hidden exprtk state
// ============================================================

struct ExpressionIC::Impl {
    using T        = double;
    using SymTab   = exprtk::symbol_table<T>;
    using Expr     = exprtk::expression<T>;
    using Parser   = exprtk::parser<T>;

    // ---- Symbol-table variables (updated per cell before evaluation) ----
    // Physics constants — set once from SimConfig.
    T gamma_v{}, beta_v{}, H_z0_v{}, r_0_v{};
    // Spatial — updated per cell.
    mutable T z_v{}, r_v{}, r_z_v{};
    // Field values — updated in evaluation order so cross-field expressions work.
    mutable T rho_v{}, v_z_v{}, v_r_v{}, v_phi_v{};
    mutable T H_z_v{}, H_r_v{}, H_phi_v{}, e_v{};

    SymTab symtab;

    // ---- One compiled expression per field ----
    struct CompiledField {
        std::string src;   ///< original expression string, kept for error messages
        Expr        expr;
    };

    CompiledField rho_f, v_z_f, v_r_f, v_phi_f;
    CompiledField H_z_f, H_r_f, H_phi_f, e_f;

    // ----------------------------------------------------------------
    // Build the shared symbol table and compile all eight expressions.
    // ----------------------------------------------------------------
    explicit Impl(const std::string& rho_s, const std::string& v_z_s,
                  const std::string& v_r_s,  const std::string& v_phi_s,
                  const std::string& H_z_s,  const std::string& H_r_s,
                  const std::string& H_phi_s, const std::string& e_s)
    {
        // ---- Register constants ----
        symtab.add_constant("pi",   M_PI);
        // (Euler's 'e' is intentionally omitted to avoid collision with the
        // specific-energy field variable 'e'.  Use exp(1) if needed.)

        // ---- Register physics constants (values set later in SetPhysics) ----
        symtab.add_variable("gamma", gamma_v);
        symtab.add_variable("beta",  beta_v);
        symtab.add_variable("H_z0",  H_z0_v);
        symtab.add_variable("r_0",   r_0_v);

        // ---- Register spatial variables ----
        symtab.add_variable("z",   z_v);
        symtab.add_variable("r",   r_v);
        symtab.add_variable("r_z", r_z_v);

        // ---- Register field variables (cross-field expressions) ----
        symtab.add_variable("rho",   rho_v);
        symtab.add_variable("v_z",   v_z_v);
        symtab.add_variable("v_r",   v_r_v);
        symtab.add_variable("v_phi", v_phi_v);
        symtab.add_variable("H_z",   H_z_v);
        symtab.add_variable("H_r",   H_r_v);
        symtab.add_variable("H_phi", H_phi_v);
        symtab.add_variable("e",     e_v);

        // ---- Compile all eight expressions ----
        Compile(rho_f,   rho_s,   "rho");
        Compile(v_z_f,   v_z_s,   "v_z");
        Compile(v_r_f,   v_r_s,   "v_r");
        Compile(v_phi_f, v_phi_s, "v_phi");
        Compile(H_z_f,   H_z_s,   "H_z");
        Compile(H_r_f,   H_r_s,   "H_r");
        Compile(H_phi_f, H_phi_s, "H_phi");
        Compile(e_f,     e_s,     "e");
    }

    // ---- Set physics constants once, before any Apply() calls ----
    void SetPhysics(double gamma, double beta, double H_z0, double r_0) {
        gamma_v = gamma;
        beta_v  = beta;
        H_z0_v  = H_z0;
        r_0_v   = r_0;
    }

    // ---- Evaluate all eight fields for one cell ----
    // Updates the mutable field variables in evaluation order so that
    // later expressions can reference earlier results (e.g. H_r = H_z * r_z).
    void EvalCell(double z, double r, double r_z) const {
        z_v   = z;
        r_v   = r;
        r_z_v = r_z;

        rho_v   = rho_f.expr.value();
        v_z_v   = v_z_f.expr.value();
        v_r_v   = v_r_f.expr.value();
        v_phi_v = v_phi_f.expr.value();
        H_z_v   = H_z_f.expr.value();
        H_r_v   = H_r_f.expr.value();
        H_phi_v = H_phi_f.expr.value();
        e_v     = e_f.expr.value();
    }

private:
    void Compile(CompiledField& cf, const std::string& expr_str, const char* field_name) {
        cf.src = expr_str;
        cf.expr.register_symbol_table(symtab);

        Parser parser;
        if (!parser.compile(expr_str, cf.expr)) {
            std::string msg = "ExpressionIC: failed to compile expression for '";
            msg += field_name;
            msg += "':\n  expression : ";
            msg += expr_str;
            msg += "\n  error      : ";
            for (std::size_t i = 0; i < parser.error_count(); ++i) {
                if (i > 0) msg += "; ";
                msg += parser.get_error(i).diagnostic;
            }
            throw std::runtime_error(msg);
        }
    }
};

// ============================================================
// YAML helpers
// ============================================================

namespace {

/// Extract the expression string for one field from the params node.
/// The node is expected to be either:
///   - absent / null  →  return the provided default expression
///   - a scalar       →  use its string value directly
///   - a map          →  error (the expression IC does not accept map-form
///                       configs; those belong to uniform_mhd)
auto ReadExpr(const YAML::Node& params, const char* key,
              const std::string& default_expr) -> std::string
{
    if (!params || params.IsNull()) return default_expr;

    const YAML::Node& n = params[key];
    if (!n || n.IsNull()) return default_expr;

    if (n.IsScalar()) {
        return n.as<std::string>();
    }

    throw std::runtime_error(
        std::string("ExpressionIC: field '") + key +
        "' must be a scalar expression string (e.g. \"H_z * r_z\").\n"
        "  For structured presets use initial_conditions type 'uniform_mhd'.");
}

} // namespace

// ============================================================
// ExpressionIC — constructor and destructor
// ============================================================

ExpressionIC::ExpressionIC(const YAML::Node& params)
{
    // Default expressions reproduce the physics-derived uniform_mhd defaults.
    const std::string rho_s   = ReadExpr(params, "rho",   "1.0");
    const std::string v_z_s   = ReadExpr(params, "v_z",   "0.0");
    const std::string v_r_s   = ReadExpr(params, "v_r",   "0.0");
    const std::string v_phi_s = ReadExpr(params, "v_phi", "0.0");
    const std::string H_z_s   = ReadExpr(params, "H_z",   "H_z0");
    const std::string H_r_s   = ReadExpr(params, "H_r",   "H_z * r_z");
    const std::string H_phi_s = ReadExpr(params, "H_phi", "(1 - 0.9 * z) * r_0 / r");
    const std::string e_s     = ReadExpr(params, "e",     "beta / (2 * (gamma - 1))");

    // Construct Impl (compiles all expressions — may throw on syntax error).
    impl_ = std::make_unique<Impl>(rho_s, v_z_s, v_r_s, v_phi_s,
                                    H_z_s, H_r_s, H_phi_s, e_s);
}

// Defined here so the compiler sees the complete Impl type for deletion.
ExpressionIC::~ExpressionIC() = default;

// ============================================================
// Apply
// ============================================================

void ExpressionIC::Apply(Fields& f, const Grid& grid,
                          const SimConfig& cfg, int l_start) const
{
    const double gamma = cfg.gamma;
    const double dz    = cfg.dz;

    // Bind physics constants into the symbol table once.
    impl_->SetPhysics(gamma, cfg.beta, cfg.H_z0, grid.r_0);

    // Note: expression evaluation uses shared mutable state in impl_.
    // Running this loop with OpenMP would require per-thread symbol tables.
    // The initialisation phase runs once, so the single-threaded cost is
    // negligible relative to the time-stepping loop.
    for (int l = 1; l < f.rows - 1; ++l) {
        const int    l_global = l_start + l - 1;
        const double z        = l_global * dz;

        for (int m = 1; m < f.cols - 1; ++m) {
            impl_->EvalCell(z, grid.r[l][m], grid.r_z[l][m]);

            f.rho  [l][m] = impl_->rho_v;
            f.v_z  [l][m] = impl_->v_z_v;
            f.v_r  [l][m] = impl_->v_r_v;
            f.v_phi[l][m] = impl_->v_phi_v;
            f.H_z  [l][m] = impl_->H_z_v;
            f.H_r  [l][m] = impl_->H_r_v;
            f.H_phi[l][m] = impl_->H_phi_v;
            f.e    [l][m] = impl_->e_v;

            // Derived scalars — consistent with Fields::UpdatePhysicalFromU.
            f.p[l][m] = (gamma - 1.0) * f.rho[l][m] * f.e[l][m];
            f.P[l][m] = f.p[l][m]
                      + 0.5 * (f.H_z  [l][m] * f.H_z  [l][m]
                               + f.H_r  [l][m] * f.H_r  [l][m]
                               + f.H_phi[l][m] * f.H_phi[l][m]);
        }
    }
}
