#pragma once
#include "types.hpp"

// Apply all boundary conditions
void ApplyBoundaryConditions(PhysicalFields& fields, ConservativeVars& u, 
                               const GridGeometry& grid, const DomainInfo& domain,
                               const SimulationParams& params, double r_0);
