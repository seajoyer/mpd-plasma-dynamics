#pragma once
#include "types.hpp"

// Initialize conservative variables from physical fields
void InitializeConservativeVars(ConservativeVars& u0, const PhysicalFields& fields,
                                  const GridGeometry& grid, int local_L_with_ghosts, int M_max);

// Compute one time step using Lax-Friedrichs scheme
void ComputeTimeStep(ConservativeVars& u, const ConservativeVars& u0,
                      const PhysicalFields& fields, const GridGeometry& grid,
                      const DomainInfo& domain, const SimulationParams& params);

// Update physical fields from conservative variables
void UpdatePhysicalFields(PhysicalFields& fields, const ConservativeVars& u,
                           const GridGeometry& grid, int local_L_with_ghosts, int M_max,
                           double gamma);

// Copy current conservative variables to u0
void CopyConservativeVars(ConservativeVars& u0, const ConservativeVars& u,
                           int local_L_with_ghosts, int M_max);
