#pragma once
#include "types.hpp"

// Apply all boundary conditions
void apply_boundary_conditions(PhysicalFields& fields, ConservativeVars& u, 
                               const GridGeometry& grid, const DomainInfo& domain,
                               const SimulationParams& params, double r_0);
