#include "ics/expression_ic.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
// Reserved symbol names — user vars may not shadow these.
// ============================================================

namespace {

const std::vector<std::string>& ReservedSymbols() {
    static const std::vector<std::string> names = {
        // physics constants
        "gamma", "beta", "H_z0", "r_0",
        // spatial
        "z", "r", "r_z",
        // math constant
        "pi",
        // field cross-references
        "rho", "v_z", "v_r", "v_phi", "H_z", "H_r", "H_phi", "e",
    };
    return names;
}

} // namespace

// ============================================================
// Impl — hidden exprtk state
// ============================================================

struct ExpressionIC::Impl {
    using T      = double;
    using SymTab = exprtk::symbol_table<T>;
    using Expr   = exprtk::expression<T>;
    using Parser = exprtk::parser<T>;

    // ---- Symbol-table variables ----
    // Physics constants — set once from SimConfig before each Apply() call.
    T gamma_v{}, beta_v{}, H_z0_v{}, r_0_v{};
    // Spatial — updated per cell before evaluation.
    mutable T z_v{}, r_v{}, r_z_v{};
    // Field values — updated in evaluation order so cross-field expressions work.
    mutable T rho_v{}, v_z_v{}, v_r_v{}, v_phi_v{};
    mutable T H_z_v{}, H_r_v{}, H_phi_v{}, e_v{};

    SymTab symtab;

    // ---- One compiled expression per physical field ----
    struct CompiledField {
        std::string src;   ///< original expression string, kept for error messages
        Expr        expr;
    };

    CompiledField rho_f, v_z_f, v_r_f, v_phi_f;
    CompiledField H_z_f, H_r_f, H_phi_f, e_f;

    // ----------------------------------------------------------------
    // Build the shared symbol table and compile all eight expressions.
    // user_vars are registered as read-only constants before compilation,
    // so they are visible in every field expression.
    // ----------------------------------------------------------------
    explicit Impl(const std::unordered_map<std::string, double>& user_vars,
                  const std::string& rho_s,   const std::string& v_z_s,
                  const std::string& v_r_s,   const std::string& v_phi_s,
                  const std::string& H_z_s,   const std::string& H_r_s,
                  const std::string& H_phi_s, const std::string& e_s)
    {
        // ---- Built-in math constant ----
        symtab.add_constant("pi", M_PI);

        // ---- Physics constants (updated once per Apply call) ----
        symtab.add_variable("gamma", gamma_v);
        symtab.add_variable("beta",  beta_v);
        symtab.add_variable("H_z0",  H_z0_v);
        symtab.add_variable("r_0",   r_0_v);

        // ---- Spatial variables (updated per cell) ----
        symtab.add_variable("z",   z_v);
        symtab.add_variable("r",   r_v);
        symtab.add_variable("r_z", r_z_v);

        // ---- Field cross-references (updated in evaluation order) ----
        symtab.add_variable("rho",   rho_v);
        symtab.add_variable("v_z",   v_z_v);
        symtab.add_variable("v_r",   v_r_v);
        symtab.add_variable("v_phi", v_phi_v);
        symtab.add_variable("H_z",   H_z_v);
        symtab.add_variable("H_r",   H_r_v);
        symtab.add_variable("H_phi", H_phi_v);
        symtab.add_variable("e",     e_v);

        // ---- User-defined constants ----
        // Validated against the reserved list before registration.
        const auto& reserved = ReservedSymbols();
        for (const auto& [name, value] : user_vars) {
            for (const auto& res : reserved) {
                if (name == res) {
                    throw std::runtime_error(
                        "ExpressionIC: user variable '" + name +
                        "' shadows the built-in symbol '" + res +
                        "'. Choose a different name.");
                }
            }
            if (!symtab.add_constant(name, value)) {
                throw std::runtime_error(
                    "ExpressionIC: failed to register user variable '" + name +
                    "' (name may already be taken by exprtk).");
            }
        }

        // ---- Compile all field expressions ----
        Compile(rho_f,   rho_s,   "rho");
        Compile(v_z_f,   v_z_s,   "v_z");
        Compile(v_r_f,   v_r_s,   "v_r");
        Compile(v_phi_f, v_phi_s, "v_phi");
        Compile(H_z_f,   H_z_s,   "H_z");
        Compile(H_r_f,   H_r_s,   "H_r");
        Compile(H_phi_f, H_phi_s, "H_phi");
        Compile(e_f,     e_s,     "e");
    }

    // ---- Bind physics constants — call once before looping over cells ----
    void SetPhysics(double gamma, double beta, double H_z0, double r_0) noexcept {
        gamma_v = gamma;
        beta_v  = beta;
        H_z0_v  = H_z0;
        r_0_v   = r_0;
    }

    // ---- Evaluate all eight fields for one cell ----
    // Updates the mutable field variables in evaluation order so that
    // later expressions can reference earlier results (e.g. H_r = H_z * r_z).
    void EvalCell(double z, double r, double r_z) const noexcept {
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
    void Compile(CompiledField& cf, const std::string& src, const char* field_name) {
        cf.src = src;
        cf.expr.register_symbol_table(symtab);

        Parser parser;
        if (!parser.compile(src, cf.expr)) {
            std::string msg;
            msg.reserve(256);
            msg += "ExpressionIC: failed to compile expression for '";
            msg += field_name;
            msg += "':\n  expression : ";
            msg += src;
            msg += "\n  errors     :";
            for (std::size_t i = 0; i < parser.error_count(); ++i) {
                msg += "\n    [";
                msg += std::to_string(i + 1);
                msg += "] ";
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
/// Accepts a YAML scalar (bare number or quoted string) or absence / null.
/// Rejects map-form nodes to catch accidental use of old preset syntax.
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
        "' must be a scalar expression string or a plain number "
        "(e.g. rho: 1.0  or  H_r: \"H_z * r_z\").\n"
        "  Map-form entries are not supported by the expression IC.");
}

/// Parse the optional params.vars block into a name→value map.
/// Every value must be a scalar convertible to double.
auto ReadUserVars(const YAML::Node& params)
    -> std::unordered_map<std::string, double>
{
    std::unordered_map<std::string, double> vars;
    if (!params || params.IsNull()) return vars;

    const YAML::Node& n = params["vars"];
    if (!n || n.IsNull()) return vars;

    if (!n.IsMap()) {
        throw std::runtime_error(
            "ExpressionIC: 'vars' must be a YAML map of name: value pairs.");
    }

    for (const auto& item : n) {
        const std::string name = item.first.as<std::string>();

        if (!item.second.IsScalar()) {
            throw std::runtime_error(
                "ExpressionIC: value for var '" + name +
                "' must be a scalar number.");
        }

        vars[name] = item.second.as<double>();
    }

    return vars;
}

} // namespace

// ============================================================
// ExpressionIC — constructor and destructor
// ============================================================

ExpressionIC::ExpressionIC(const YAML::Node& params)
{
    // ---- 1. User-defined constants (parsed first so they can be
    //         validated against reserved names before compilation) ----
    const auto user_vars = ReadUserVars(params);

    // ---- 2. Field expressions (physics-derived defaults match the
    //         standard MHD initialisation used before this IC existed) ----
    const std::string rho_s   = ReadExpr(params, "rho",   "1.0");
    const std::string v_z_s   = ReadExpr(params, "v_z",   "0.0");
    const std::string v_r_s   = ReadExpr(params, "v_r",   "0.0");
    const std::string v_phi_s = ReadExpr(params, "v_phi", "0.0");
    const std::string H_z_s   = ReadExpr(params, "H_z",   "H_z0");
    const std::string H_r_s   = ReadExpr(params, "H_r",   "H_z * r_z");
    const std::string H_phi_s = ReadExpr(params, "H_phi", "(1 - 0.9 * z) * r_0 / r");
    const std::string e_s     = ReadExpr(params, "e",     "beta / (2 * (gamma - 1))");

    // ---- 3. Construct Impl (registers symbols, compiles expressions) ----
    // Construction may throw on syntax errors — the caller gets a clear message.
    impl_ = std::make_unique<Impl>(user_vars,
                                   rho_s, v_z_s, v_r_s, v_phi_s,
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

    // Bind physics constants into the shared symbol table once.
    impl_->SetPhysics(gamma, cfg.beta, cfg.H_z0, grid.r_0);

    // Expression evaluation uses shared mutable state in impl_.
    // Running this loop with OpenMP would require per-thread symbol tables,
    // which exprtk does not support out of the box.  The initialisation phase
    // runs exactly once before time-stepping, so the single-threaded cost is
    // negligible in practice.
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

            // Derived scalars — must follow all primary assignments.
            f.p[l][m] = (gamma - 1.0) * f.rho[l][m] * f.e[l][m];
            f.P[l][m] = f.p[l][m]
                      + 0.5 * (f.H_z  [l][m] * f.H_z  [l][m]
                               + f.H_r  [l][m] * f.H_r  [l][m]
                               + f.H_phi[l][m] * f.H_phi[l][m]);
        }
    }
}
