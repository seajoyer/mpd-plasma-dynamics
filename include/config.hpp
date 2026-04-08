#pragma once

#include <string>
#include <vector>

#include "field_condition.hpp"

// ============================================================
// Geometry configuration
// ============================================================

/// Runtime configuration for the channel geometry.
/// `type`        maps to a name registered in GeometryRegistry.
/// `params_yaml` holds the raw YAML text of the optional `params` sub-node,
///               which is parsed and forwarded to the geometry factory.
struct GeometryConfig {
    std::string type        = "coaxial_nozzle";
    std::string params_yaml;   ///< empty → no params; factory receives a null node
};

// ============================================================
// Boundary-condition configuration
// ============================================================

/// Per-field boundary-condition specification for one contiguous segment of a face.
///
/// global_lo / global_hi are indices along the face's *free* axis in global
/// coordinates (l for M faces, m for L faces).  A negative value is a
/// sentinel meaning "use the face's natural start/end":
///
///   global_lo < 0  →  0                        (first cell on the face)
///   global_hi < 0  →  L_max_global-1 or M_max  (last cell on the face)
///
/// Each physical field has an independent FieldCond.  Unspecified fields
/// default to Neumann (zero-gradient).
///
/// Example YAML for a solid inner wall:
///
///   m_lo:
///     - range: [0, 260]
///       rho:   neumann
///       v_z:   neumann
///       v_r:   wall_tangent
///       v_phi: { dirichlet: 0.0 }
///       e:     neumann
///       H_z:   neumann
///       H_r:   wall_tangent
///       H_phi: neumann
///
/// Example YAML for the axis-of-symmetry region:
///
///   m_lo:
///     - range: [261, -1]
///       rho:   axis_lf
///       v_z:   axis_lf
///       v_r:   { dirichlet: 0.0 }
///       v_phi: { dirichlet: 0.0 }
///       e:     axis_lf
///       H_z:   axis_lf
///       H_r:   { dirichlet: 0.0 }
///       H_phi: { dirichlet: 0.0 }
struct BCSegmentConfig {
    int global_lo = -1;   ///< negative → face start
    int global_hi = -1;   ///< negative → face end

    // Per-field conditions — all default to Neumann.
    FieldCond rho;
    FieldCond v_z;
    FieldCond v_r;
    FieldCond v_phi;
    FieldCond e;
    FieldCond H_z;
    FieldCond H_r;
    FieldCond H_phi;
};

/// All segments for one face (l_lo, l_hi, m_lo, or m_hi).
struct BCFaceConfig {
    std::vector<BCSegmentConfig> segments;
};

// ============================================================
// SimConfig
// ============================================================

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
    double dt = 0.000025;   ///< initial (and fixed) time step when adaptive_dt = false

    // ---- adaptive time step -----------------------------------------------
    bool   adaptive_dt      = false;
    double cfl_number       = 0.5;
    double dt_growth_factor = 1.1;
    double dt_min           = 1.0e-9;
    double dt_max           = 1.0e-3;

    // ---- global grid ----
    int L_max_global = 800;
    int M_max        = 400;

    // ---- convergence ----
    double convergence_threshold = 0.0;   ///< 0 → disabled
    int    check_frequency       = 100;

    // ---- output ----
    std::string output_dir = "output";
    std::string run_name   = "run";
    int         vtk_step   = 100;

    // ---- parallelism ----
    int openmp_threads = 0;
    int mpi_dims_l     = 0;
    int mpi_dims_m     = 0;

    // ---- geometry ----
    GeometryConfig geometry;

    // ---- boundary conditions (one BCFaceConfig per Cartesian face) ----
    BCFaceConfig bc_l_lo;   ///< z = 0  face (inflow by default)
    BCFaceConfig bc_l_hi;   ///< z = L  face (outflow by default)
    BCFaceConfig bc_m_lo;   ///< r = inner face (wall + axis by default)
    BCFaceConfig bc_m_hi;   ///< r = outer face (wall by default)

    // ---- derived (computed by load / init) ----
    double dz{};
    double dy{};

    SimConfig() { init(); }

    /// Load all parameters from a YAML file and compute derived quantities.
    void load(const std::string& path = "config.yaml");

private:
    void init();
};
