#pragma once
#include "types.hpp"

// Initialize conservative variables from physical fields (2D version)
void InitializeConservativeVars2D(ConservativeVars& u0, const PhysicalFields& fields,
                                   const GridGeometry& grid, const DomainInfo& domain);

// Compute one time step using Lax-Friedrichs scheme (2D version)
void ComputeTimeStep2D(ConservativeVars& u, const ConservativeVars& u0,
                        const PhysicalFields& fields, const GridGeometry& grid,
                        const DomainInfo& domain, const SimulationParams& params);

// Update physical fields from conservative variables (2D version)
void UpdatePhysicalFields2D(PhysicalFields& fields, const ConservativeVars& u,
                             const GridGeometry& grid, const DomainInfo& domain,
                             double gamma);

// Copy current conservative variables to u0 (2D version)
void CopyConservativeVars2D(ConservativeVars& u0, const ConservativeVars& u,
                             const DomainInfo& domain);

// Legacy 1D versions (kept for compatibility)
void InitializeConservativeVars(ConservativeVars& u0, const PhysicalFields& fields,
                                  const GridGeometry& grid, int local_L_with_ghosts, int M_max);

void ComputeTimeStep(ConservativeVars& u, const ConservativeVars& u0,
                      const PhysicalFields& fields, const GridGeometry& grid,
                      const DomainInfo& domain, const SimulationParams& params);

void UpdatePhysicalFields(PhysicalFields& fields, const ConservativeVars& u,
                           const GridGeometry& grid, int local_L_with_ghosts, int M_max,
                           double gamma);

void CopyConservativeVars(ConservativeVars& u0, const ConservativeVars& u,
                           int local_L_with_ghosts, int M_max);
