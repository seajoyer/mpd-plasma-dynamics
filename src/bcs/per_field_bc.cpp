// exprtk must be included before any Anthropic headers because it defines
// macros that must match across every translation unit that includes it.
#define exprtk_disable_string_capabilities
#define exprtk_disable_rtl_io_file
#define exprtk_disable_rtl_vecops
#include <exprtk.hpp>

#include "bcs/per_field_bc.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "bc_context.hpp"
#include "physics_utils.hpp"

// ============================================================
// ExprImpl — compiled per-cell boundary expressions
//
// One instance is shared by all calls to PerFieldBC::Apply().
// The mutable symbol-table variables are written before each
// cell evaluation so the expression sees the correct context.
//
// Cross-field references work within a single cell: after each
// field is evaluated its result is written back to the
// corresponding symbol-table variable, so expressions for later
// fields in the evaluation order can reference earlier ones.
//
// Forward references (a field referencing one that comes later
// in the order) see whatever value was left over from the
// previous cell — typically meaningless.  Users should not rely
// on forward references.
// ============================================================

struct PerFieldBC::ExprImpl {
    using T      = double;
    using SymTab = exprtk::symbol_table<T>;
    using Expr   = exprtk::expression<T>;
    using Parser = exprtk::parser<T>;

    // ---- Symbol-table variables ----------------------------------------
    // Physics constants — set once per Apply() call.
    T gamma_v{}, beta_v{}, H_z0_v{}, r_0_v{};
    T dz_v{}, dr_v{};

    // Spatial at the boundary cell — set per cell.
    mutable T r_v{}, r_z_v{}, z_v{};

    // Spatial at the interior neighbour — set per cell.
    mutable T r_nb_v{}, r_z_nb_v{};

    // Interior-neighbour field values — set per cell.
    mutable T rho_nb_v{}, v_z_nb_v{}, v_r_nb_v{}, v_phi_nb_v{};
    mutable T e_nb_v{}, H_z_nb_v{}, H_r_nb_v{}, H_phi_nb_v{};

    // Boundary-cell values updated in evaluation order so later fields
    // can reference earlier ones within the same cell.
    mutable T rho_v{}, v_z_v{}, v_r_v{}, v_phi_v{};
    mutable T H_z_v{}, H_r_v{}, H_phi_v{}, e_v{};

    SymTab symtab;

    struct CompiledField {
        std::string src;
        Expr        expr;
        bool        active{false};
    };

    // One entry per physical field, in evaluation order.
    CompiledField cf_rho, cf_v_z, cf_v_r, cf_v_phi;
    CompiledField cf_H_z, cf_H_r, cf_H_phi, cf_e;

    // ----------------------------------------------------------------
    // Constructor: register all symbols and compile active expressions.
    // ----------------------------------------------------------------
    ExprImpl(const FieldCond& fc_rho,   const FieldCond& fc_v_z,
             const FieldCond& fc_v_r,   const FieldCond& fc_v_phi,
             const FieldCond& fc_H_z,   const FieldCond& fc_H_r,
             const FieldCond& fc_H_phi, const FieldCond& fc_e)
    {
        // ---- Physics constants ----
        symtab.add_variable("gamma", gamma_v);
        symtab.add_variable("beta",  beta_v);
        symtab.add_variable("H_z0",  H_z0_v);
        symtab.add_variable("r_0",   r_0_v);
        symtab.add_variable("dz",    dz_v);
        symtab.add_variable("dr",    dr_v);

        // ---- Spatial at boundary cell ----
        symtab.add_variable("r",   r_v);
        symtab.add_variable("r_z", r_z_v);
        symtab.add_variable("z",   z_v);

        // ---- Spatial at interior neighbour ----
        symtab.add_variable("r_nb",   r_nb_v);
        symtab.add_variable("r_z_nb", r_z_nb_v);

        // ---- Math constant ----
        symtab.add_constant("pi", M_PI);

        // ---- Interior-neighbour field values ----
        symtab.add_variable("rho_nb",   rho_nb_v);
        symtab.add_variable("v_z_nb",   v_z_nb_v);
        symtab.add_variable("v_r_nb",   v_r_nb_v);
        symtab.add_variable("v_phi_nb", v_phi_nb_v);
        symtab.add_variable("e_nb",     e_nb_v);
        symtab.add_variable("H_z_nb",   H_z_nb_v);
        symtab.add_variable("H_r_nb",   H_r_nb_v);
        symtab.add_variable("H_phi_nb", H_phi_nb_v);

        // ---- Current boundary values (updated in evaluation order) ----
        symtab.add_variable("rho",   rho_v);
        symtab.add_variable("v_z",   v_z_v);
        symtab.add_variable("v_r",   v_r_v);
        symtab.add_variable("v_phi", v_phi_v);
        symtab.add_variable("H_z",   H_z_v);
        symtab.add_variable("H_r",   H_r_v);
        symtab.add_variable("H_phi", H_phi_v);
        symtab.add_variable("e",     e_v);

        // ---- Compile active expressions in evaluation order ----
        Compile(cf_rho,   fc_rho,   "rho");
        Compile(cf_v_z,   fc_v_z,   "v_z");
        Compile(cf_v_r,   fc_v_r,   "v_r");
        Compile(cf_v_phi, fc_v_phi, "v_phi");
        Compile(cf_H_z,   fc_H_z,   "H_z");
        Compile(cf_H_r,   fc_H_r,   "H_r");
        Compile(cf_H_phi, fc_H_phi, "H_phi");
        Compile(cf_e,     fc_e,     "e");
    }

    // Set physics constants (call once at the top of Apply()).
    void SetPhysics(double gamma, double beta, double H_z0, double r_0,
                    double dz, double dr) noexcept {
        gamma_v = gamma; beta_v = beta; H_z0_v = H_z0; r_0_v = r_0;
        dz_v = dz; dr_v = dr;
    }

    // Update spatial symbols for the current cell (boundary + neighbour).
    void SetSpatial(double r, double r_z, double z,
                    double r_nb, double r_z_nb) const noexcept {
        r_v = r; r_z_v = r_z; z_v = z;
        r_nb_v = r_nb; r_z_nb_v = r_z_nb;
    }

    // Update neighbour field values for the current cell.
    void SetNeighbors(double rho_nb, double v_z_nb, double v_r_nb, double v_phi_nb,
                      double e_nb,   double H_z_nb,  double H_r_nb,  double H_phi_nb)
        const noexcept
    {
        rho_nb_v   = rho_nb;   v_z_nb_v  = v_z_nb;  v_r_nb_v  = v_r_nb;
        v_phi_nb_v = v_phi_nb; e_nb_v    = e_nb;
        H_z_nb_v   = H_z_nb;   H_r_nb_v  = H_r_nb;  H_phi_nb_v = H_phi_nb;
    }

private:
    void Compile(CompiledField& cf, const FieldCond& fc, const char* field_name) {
        if (fc.type != FieldCondType::Expression) return;

        cf.active = true;
        cf.src    = fc.expr_str;
        cf.expr.register_symbol_table(symtab);

        Parser parser;
        if (!parser.compile(cf.src, cf.expr)) {
            std::string msg;
            msg.reserve(512);
            msg += "PerFieldBC: failed to compile BC expression for field '";
            msg += field_name;
            msg += "':\n  expression : ";
            msg += cf.src;
            msg += "\n  error(s)   :";
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
// Constructor
// ============================================================

PerFieldBC::PerFieldBC(enum FaceBC::Face face, const BCSegmentConfig& seg)
    : face_(face),
      rho_  (seg.rho),
      v_z_  (seg.v_z),
      v_r_  (seg.v_r),
      v_phi_(seg.v_phi),
      e_    (seg.e),
      H_z_  (seg.H_z),
      H_r_  (seg.H_r),
      H_phi_(seg.H_phi),
      has_axis_lf_(seg.rho.type   == FieldCondType::AxisLF ||
                   seg.v_z.type   == FieldCondType::AxisLF ||
                   seg.e.type     == FieldCondType::AxisLF ||
                   seg.H_z.type   == FieldCondType::AxisLF),
      has_expressions_(seg.rho.type   == FieldCondType::Expression ||
                       seg.v_z.type   == FieldCondType::Expression ||
                       seg.v_r.type   == FieldCondType::Expression ||
                       seg.v_phi.type == FieldCondType::Expression ||
                       seg.e.type     == FieldCondType::Expression ||
                       seg.H_z.type   == FieldCondType::Expression ||
                       seg.H_r.type   == FieldCondType::Expression ||
                       seg.H_phi.type == FieldCondType::Expression)
{
    // ---- Validate AxisLF placement ----
    if (has_axis_lf_ && face_ != FaceBC::Face::M_LO) {
        throw std::runtime_error(
            "PerFieldBC: AxisLF condition is only valid on the M_LO face "
            "(inner boundary / axis of symmetry)");
    }

    // ---- Validate AxisLF field selection ----
    // AxisLF is only defined for rho, v_z, e, H_z (mapped to u_1, u_2, u_5, u_7).
    // The remaining four fields have no half-stencil LF formula.
    const auto require_no_axis_lf = [](const FieldCond& fc, const char* name) -> void {
        if (fc.type == FieldCondType::AxisLF) {
            throw std::runtime_error(
                std::string("PerFieldBC: AxisLF is only valid for rho, v_z, e, and H_z "
                            "(fields with a defined half-stencil LF update).  '") +
                name + "' has no AxisLF formula.  "
                "Use { dirichlet: 0.0 } for symmetry components that must be zero.");
        }
    };
    require_no_axis_lf(seg.v_r,   "v_r");
    require_no_axis_lf(seg.v_phi, "v_phi");
    require_no_axis_lf(seg.H_r,   "H_r");
    require_no_axis_lf(seg.H_phi, "H_phi");

    // ---- Build expression engine if needed ----
    if (has_expressions_) {
        // ExprImpl constructor compiles all Expression-typed fields and throws
        // std::runtime_error with a descriptive message on any syntax error.
        expr_impl_ = std::make_unique<ExprImpl>(
            seg.rho, seg.v_z, seg.v_r, seg.v_phi,
            seg.H_z, seg.H_r, seg.H_phi, seg.e);
    }
}

// Destructor defined here so the compiler sees the complete ExprImpl type.
PerFieldBC::~PerFieldBC() = default;

// ============================================================
// Apply
// ============================================================

void PerFieldBC::Apply(BCContext& ctx) const {
    Fields&          f   = ctx.fields;
    const Grid&      g   = ctx.grid;
    const SimConfig& cfg = ctx.cfg;
    const MPIManager& mpi = ctx.mpi;
    const double dt = ctx.dt;
    const double dz = cfg.dz;

    const bool is_l_face = (face_ == FaceBC::Face::L_LO ||
                            face_ == FaceBC::Face::L_HI);

    // ---- Fixed index and interior-neighbour index (fixed axis) ----
    int l_fix{}, m_fix{}, l_nb{}, m_nb{};
    switch (face_) {
        case FaceBC::Face::L_LO:
            l_fix = 1;            l_nb = 2;                 break;
        case FaceBC::Face::L_HI:
            l_fix = mpi.local_L;  l_nb = mpi.local_L - 1;  break;
        case FaceBC::Face::M_LO:
            m_fix = 1;            m_nb = 2;                 break;
        case FaceBC::Face::M_HI:
            m_fix = mpi.local_M;  m_nb = mpi.local_M - 1;  break;
    }

    // ---- Initialise expression physics constants (once per call) ----
    // dr varies per l-index for M faces; it is set inside the loop below.
    if (expr_impl_) {
        expr_impl_->SetPhysics(cfg.gamma, cfg.beta, cfg.H_z0, g.r_0, dz, 0.0);
    }

    // ================================================================
    //  Helper: resolve one scalar condition (Neumann / Dirichlet).
    //  Expression and AxisLF are handled explicitly in the cell loops.
    // ================================================================
    auto resolve = [](const FieldCond& c, double cell, double nb) -> double {
        switch (c.type) {
            case FieldCondType::Neumann:   return nb + c.value;
            case FieldCondType::Dirichlet: return c.value;
            default:                       return cell;  // Expression / AxisLF: caller handles
        }
    };

    // ================================================================
    //  Macro-like helper (lambda) to evaluate one field and update the
    //  expression cross-reference variable.
    //  Used in BOTH the expression and simple paths to avoid duplication.
    // ================================================================

    // ================================================================
    //  L faces — iterate over m, fixed l
    // ================================================================
    if (is_l_face) {
        const int l  = l_fix;
        const int ln = l_nb;
        const int l_global = mpi.l_start + l - 1;
        const double z_l   = l_global * dz;

        if (has_expressions_) {
            // ---- Expression path (no OpenMP: shared exprtk state) ----
            for (int m = ctx.local_lo; m <= ctx.local_hi; ++m) {
                expr_impl_->dr_v = g.dr[l];
                expr_impl_->SetSpatial(g.r[l][m],  g.r_z[l][m],  z_l,
                                       g.r[ln][m], g.r_z[ln][m]);
                expr_impl_->SetNeighbors(
                    f.rho[ln][m],   f.v_z[ln][m],   f.v_r[ln][m],   f.v_phi[ln][m],
                    f.e[ln][m],     f.H_z[ln][m],   f.H_r[ln][m],   f.H_phi[ln][m]);

                // Evaluate and update cross-reference vars in order.
                double val;

                val = (rho_.type == FieldCondType::Expression)
                          ? expr_impl_->cf_rho.expr.value()
                          : resolve(rho_, f.rho[l][m], f.rho[ln][m]);
                f.rho[l][m] = val;   expr_impl_->rho_v = val;

                val = (v_z_.type == FieldCondType::Expression)
                          ? expr_impl_->cf_v_z.expr.value()
                          : resolve(v_z_, f.v_z[l][m], f.v_z[ln][m]);
                f.v_z[l][m] = val;   expr_impl_->v_z_v = val;

                val = (v_r_.type == FieldCondType::Expression)
                          ? expr_impl_->cf_v_r.expr.value()
                          : resolve(v_r_, f.v_r[l][m], f.v_r[ln][m]);
                f.v_r[l][m] = val;   expr_impl_->v_r_v = val;

                val = (v_phi_.type == FieldCondType::Expression)
                          ? expr_impl_->cf_v_phi.expr.value()
                          : resolve(v_phi_, f.v_phi[l][m], f.v_phi[ln][m]);
                f.v_phi[l][m] = val; expr_impl_->v_phi_v = val;

                val = (H_z_.type == FieldCondType::Expression)
                          ? expr_impl_->cf_H_z.expr.value()
                          : resolve(H_z_, f.H_z[l][m], f.H_z[ln][m]);
                f.H_z[l][m] = val;   expr_impl_->H_z_v = val;

                val = (H_r_.type == FieldCondType::Expression)
                          ? expr_impl_->cf_H_r.expr.value()
                          : resolve(H_r_, f.H_r[l][m], f.H_r[ln][m]);
                f.H_r[l][m] = val;   expr_impl_->H_r_v = val;

                val = (H_phi_.type == FieldCondType::Expression)
                          ? expr_impl_->cf_H_phi.expr.value()
                          : resolve(H_phi_, f.H_phi[l][m], f.H_phi[ln][m]);
                f.H_phi[l][m] = val; expr_impl_->H_phi_v = val;

                val = (e_.type == FieldCondType::Expression)
                          ? expr_impl_->cf_e.expr.value()
                          : resolve(e_, f.e[l][m], f.e[ln][m]);
                f.e[l][m] = val;     expr_impl_->e_v = val;

                RebuildUFromPhysical(f, g, l, m);
                // Note: AxisLF is not valid on L faces (validated in constructor).
            }
        } else {
            // ---- Simple path (Neumann / Dirichlet only, OpenMP safe) ----
            #pragma omp parallel for
            for (int m = ctx.local_lo; m <= ctx.local_hi; ++m) {
                f.rho[l][m]   = resolve(rho_,   f.rho[l][m],   f.rho[ln][m]);
                f.v_z[l][m]   = resolve(v_z_,   f.v_z[l][m],   f.v_z[ln][m]);
                f.v_r[l][m]   = resolve(v_r_,   f.v_r[l][m],   f.v_r[ln][m]);
                f.v_phi[l][m] = resolve(v_phi_,  f.v_phi[l][m], f.v_phi[ln][m]);
                f.e[l][m]     = resolve(e_,      f.e[l][m],     f.e[ln][m]);
                f.H_z[l][m]   = resolve(H_z_,   f.H_z[l][m],   f.H_z[ln][m]);
                f.H_r[l][m]   = resolve(H_r_,   f.H_r[l][m],   f.H_r[ln][m]);
                f.H_phi[l][m] = resolve(H_phi_,  f.H_phi[l][m], f.H_phi[ln][m]);
                RebuildUFromPhysical(f, g, l, m);
            }
        }
        return;
    }

    // ================================================================
    //  M faces — iterate over l, fixed m
    // ================================================================

    const int m  = m_fix;
    const int mn = m_nb;
    const bool is_lo = (face_ == FaceBC::Face::M_LO);

    if (has_expressions_) {
        // ---- Expression path (no OpenMP: shared exprtk state) ----
        for (int l = ctx.local_lo; l <= ctx.local_hi; ++l) {
            const int    l_global = mpi.l_start + l - 1;
            const double z_l      = l_global * dz;
            const double dr_l     = g.dr[l];

            expr_impl_->dr_v = dr_l;
            expr_impl_->SetSpatial(g.r[l][m],  g.r_z[l][m],  z_l,
                                   g.r[l][mn], g.r_z[l][mn]);
            expr_impl_->SetNeighbors(
                f.rho[l][mn],   f.v_z[l][mn],   f.v_r[l][mn],   f.v_phi[l][mn],
                f.e[l][mn],     f.H_z[l][mn],   f.H_r[l][mn],   f.H_phi[l][mn]);

            double val;

            val = (rho_.type == FieldCondType::Expression)
                      ? expr_impl_->cf_rho.expr.value()
                      : resolve(rho_, f.rho[l][m], f.rho[l][mn]);
            f.rho[l][m] = val;   expr_impl_->rho_v = val;

            val = (v_z_.type == FieldCondType::Expression)
                      ? expr_impl_->cf_v_z.expr.value()
                      : resolve(v_z_, f.v_z[l][m], f.v_z[l][mn]);
            f.v_z[l][m] = val;   expr_impl_->v_z_v = val;

            val = (v_r_.type == FieldCondType::Expression)
                      ? expr_impl_->cf_v_r.expr.value()
                      : resolve(v_r_, f.v_r[l][m], f.v_r[l][mn]);
            f.v_r[l][m] = val;   expr_impl_->v_r_v = val;

            val = (v_phi_.type == FieldCondType::Expression)
                      ? expr_impl_->cf_v_phi.expr.value()
                      : resolve(v_phi_, f.v_phi[l][m], f.v_phi[l][mn]);
            f.v_phi[l][m] = val; expr_impl_->v_phi_v = val;

            val = (H_z_.type == FieldCondType::Expression)
                      ? expr_impl_->cf_H_z.expr.value()
                      : resolve(H_z_, f.H_z[l][m], f.H_z[l][mn]);
            f.H_z[l][m] = val;   expr_impl_->H_z_v = val;

            val = (H_r_.type == FieldCondType::Expression)
                      ? expr_impl_->cf_H_r.expr.value()
                      : resolve(H_r_, f.H_r[l][m], f.H_r[l][mn]);
            f.H_r[l][m] = val;   expr_impl_->H_r_v = val;

            val = (H_phi_.type == FieldCondType::Expression)
                      ? expr_impl_->cf_H_phi.expr.value()
                      : resolve(H_phi_, f.H_phi[l][m], f.H_phi[l][mn]);
            f.H_phi[l][m] = val; expr_impl_->H_phi_v = val;

            val = (e_.type == FieldCondType::Expression)
                      ? expr_impl_->cf_e.expr.value()
                      : resolve(e_, f.e[l][m], f.e[l][mn]);
            f.e[l][m] = val;     expr_impl_->e_v = val;

            // Step 2: rebuild conservative vars.
            RebuildUFromPhysical(f, g, l, m);

            // Step 3: AxisLF overwrites (expressions and AxisLF can coexist).
            if (has_axis_lf_) {
                if (rho_.type == FieldCondType::AxisLF) f.u_1[l][m] = AxisLfU1(f, g, l, m, dt, dz);
                if (v_z_.type == FieldCondType::AxisLF) f.u_2[l][m] = AxisLfU2(f, g, l, m, dt, dz);
                if (e_.type   == FieldCondType::AxisLF) f.u_5[l][m] = AxisLfU5(f, g, l, m, dt, dz);
                if (H_z_.type == FieldCondType::AxisLF) f.u_7[l][m] = AxisLfU7(f, g, l, m, dt, dz);
            }
        }
    } else {
        // ---- Simple path (Neumann / Dirichlet / AxisLF, OpenMP safe) ----
        #pragma omp parallel for
        for (int l = ctx.local_lo; l <= ctx.local_hi; ++l) {
            // Step 1: apply physical conditions.
            f.rho[l][m]   = resolve(rho_,   f.rho[l][m],   f.rho[l][mn]);
            f.v_phi[l][m] = resolve(v_phi_,  f.v_phi[l][m], f.v_phi[l][mn]);
            f.e[l][m]     = resolve(e_,      f.e[l][m],     f.e[l][mn]);
            f.H_phi[l][m] = resolve(H_phi_,  f.H_phi[l][m], f.H_phi[l][mn]);
            f.v_z[l][m]   = resolve(v_z_,   f.v_z[l][m],   f.v_z[l][mn]);
            f.H_z[l][m]   = resolve(H_z_,   f.H_z[l][m],   f.H_z[l][mn]);
            f.v_r[l][m]   = resolve(v_r_,   f.v_r[l][m],   f.v_r[l][mn]);
            f.H_r[l][m]   = resolve(H_r_,   f.H_r[l][m],   f.H_r[l][mn]);

            // Step 2: rebuild conservative vars.
            RebuildUFromPhysical(f, g, l, m);

            // Step 3: AxisLF overwrites (only on M_LO).
            if (has_axis_lf_) {
                if (rho_.type == FieldCondType::AxisLF) f.u_1[l][m] = AxisLfU1(f, g, l, m, dt, dz);
                if (v_z_.type == FieldCondType::AxisLF) f.u_2[l][m] = AxisLfU2(f, g, l, m, dt, dz);
                if (e_.type   == FieldCondType::AxisLF) f.u_5[l][m] = AxisLfU5(f, g, l, m, dt, dz);
                if (H_z_.type == FieldCondType::AxisLF) f.u_7[l][m] = AxisLfU7(f, g, l, m, dt, dz);
            }
        }
    }
}

// ============================================================
// AxisLF stencil helpers
// ============================================================

auto PerFieldBC::AxisLfU1(const Fields& f, const Grid& g, int l, int m, double dt,
                           double dz) -> double {
    auto** u0 = f.u0_1.Raw();
    auto** vz = f.v_z.Raw();
    auto** vr = f.v_r.Raw();
    auto** r  = g.r.Raw();
    const double dr_l = g.dr[l];

    return (0.25 * (u0[l+1][m]/r[l+1][m] + u0[l-1][m]/r[l-1][m]
                  + u0[l][m+1]/r[l][m+1] + u0[l][m]  /r[l][m])
            + dt * (-(u0[l+1][m]/r[l+1][m]*vz[l+1][m]
                     -u0[l-1][m]/r[l-1][m]*vz[l-1][m]) / (2.0*dz)
                   -(u0[l][m+1]/r[l][m+1]*vr[l][m+1]
                    -u0[l][m]  /r[l][m]  *vr[l][m+1]) / dr_l))
           * r[l][m];
}

auto PerFieldBC::AxisLfU2(const Fields& f, const Grid& g, int l, int m, double dt,
                           double dz) -> double {
    auto** u0 = f.u0_2.Raw();
    auto** vz = f.v_z.Raw();
    auto** vr = f.v_r.Raw();
    auto** Hz = f.H_z.Raw();
    auto** Hr = f.H_r.Raw();
    auto** P  = f.P.Raw();
    auto** r  = g.r.Raw();
    const double dr_l = g.dr[l];

    return (0.25 * (u0[l+1][m]/r[l+1][m] + u0[l-1][m]/r[l-1][m]
                  + u0[l][m+1]/r[l][m+1] + u0[l][m]  /r[l][m])
            + dt * (((Hz[l+1][m]*Hz[l+1][m] - P[l+1][m])
                    -(Hz[l-1][m]*Hz[l-1][m] - P[l-1][m])) / (2.0*dz)
                   +(Hz[l][m+1]*Hr[l][m+1] - Hz[l][m]*Hr[l][m]) / dr_l
                   -(u0[l+1][m]/r[l+1][m]*vz[l+1][m]
                    -u0[l-1][m]/r[l-1][m]*vz[l-1][m]) / (2.0*dz)
                   -(u0[l][m+1]/r[l][m+1]*vr[l][m+1]
                    -u0[l][m]  /r[l][m]  *vr[l][m])   / dr_l))
           * r[l][m];
}

auto PerFieldBC::AxisLfU5(const Fields& f, const Grid& g, int l, int m, double dt,
                           double dz) -> double {
    auto** u0 = f.u0_5.Raw();
    auto** vz = f.v_z.Raw();
    auto** vr = f.v_r.Raw();
    auto** p  = f.p.Raw();
    auto** r  = g.r.Raw();
    const double dr_l = g.dr[l];

    return (0.25 * (u0[l+1][m]/r[l+1][m] + u0[l-1][m]/r[l-1][m]
                  + u0[l][m+1]/r[l][m+1] + u0[l][m]  /r[l][m])
            + dt * (-p[l][m] * ((vz[l+1][m]-vz[l-1][m]) / (2.0*dz)
                               +(vr[l][m+1]-vr[l][m])    / dr_l)
                   -(u0[l+1][m]/r[l+1][m]*vz[l+1][m]
                    -u0[l-1][m]/r[l-1][m]*vz[l-1][m]) / (2.0*dz)
                   -(u0[l][m+1]/r[l][m+1]*vr[l][m+1]
                    -u0[l][m]  /r[l][m]  *vr[l][m])   / dr_l))
           * r[l][m];
}

auto PerFieldBC::AxisLfU7(const Fields& f, const Grid& g, int l, int m, double dt,
                           double dz) -> double {
    auto** u0 = f.u0_7.Raw();
    auto** vz = f.v_z.Raw();
    auto** vr = f.v_r.Raw();
    auto** Hz = f.H_z.Raw();
    auto** Hr = f.H_r.Raw();
    auto** r  = g.r.Raw();
    const double dr_l = g.dr[l];

    return (0.25 * (u0[l+1][m]/r[l+1][m] + u0[l-1][m]/r[l-1][m]
                  + u0[l][m+1]/r[l][m+1] + u0[l][m]  /r[l][m])
            + dt * ((Hr[l][m+1]*vz[l][m+1] - Hr[l][m]*vz[l][m]) / dr_l
                   -(u0[l][m+1]/r[l][m+1]*vr[l][m+1]
                    -u0[l][m]  /r[l][m]  *vr[l][m])              / dr_l))
           * r[l][m];
}
