#pragma once

#include <string>

class Fields;
class Grid;
struct SimConfig;

/// Abstract interface for an initial-condition implementation.
///
/// An IInitialCondition sets all physical fields (rho, v_z, v_r, v_phi,
/// e, p, P, H_z, H_r, H_phi) for the interior cells owned by this rank.
/// Ghost cells are left at their default-constructed zero values; they will
/// be populated by the first ghost-exchange in Solver::advance().
///
/// Implementations are registered by name in InitialConditionRegistry and
/// selected at runtime via the `initial_conditions.type` field in config.yaml.
///
/// Lifetime requirement: the IC object must outlive every Fields::InitPhysical
/// call that uses it (both are typically transient in main()).
class IInitialCondition {
public:
    virtual ~IInitialCondition() = default;

    /// Initialise all physical fields for interior cells [1..local_L][1..local_M].
    ///
    /// Derived fields (p, P) must also be filled so the solver has a consistent
    /// state before the first ghost exchange and central update.
    ///
    /// @param fields   Field arrays to write (physical vars: rho, v_*, H_*, e, p, P).
    /// @param grid     Local mesh (r, r_z, dr, r_0 are all available).
    /// @param cfg      Global simulation parameters (gamma, beta, H_z0, dz, ...).
    /// @param l_start  Global l-index of the first owned interior cell (local l = 1).
    ///                 Required to compute z = l_global * cfg.dz for spatial profiles.
    virtual void Apply(Fields& fields, const Grid& grid,
                       const SimConfig& cfg, int l_start) const = 0;

    [[nodiscard]] virtual auto Name() const -> std::string = 0;
};
