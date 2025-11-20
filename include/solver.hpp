#pragma once
#include "types.hpp"

// Initialize conservative variables from physical fields
void initialize_conservative_vars(ConservativeVars& u0, const PhysicalFields& fields,
                                  const GridGeometry& grid, int local_L_with_ghosts, int M_max);

// Compute one time step using Lax-Friedrichs scheme
void compute_time_step(ConservativeVars& u, const ConservativeVars& u0,
                      const PhysicalFields& fields, const GridGeometry& grid,
                      const DomainInfo& domain, const SimulationParams& params);

// Update physical fields from conservative variables
void update_physical_fields(PhysicalFields& fields, const ConservativeVars& u,
                           const GridGeometry& grid, int local_L_with_ghosts, int M_max,
                           double gamma);

// Copy current conservative variables to u0
void copy_conservative_vars(ConservativeVars& u0, const ConservativeVars& u,
                           int local_L_with_ghosts, int M_max);
