#pragma once
#include "types.hpp"

/**
 * @brief Apply all boundary conditions for 2D decomposed domain
 * 
 * Applies boundary conditions based on the process's position in the 2D grid:
 * - Left boundary (z=0, inlet): only processes with is_left_boundary
 * - Right boundary (z=z_max, outlet): only processes with is_right_boundary
 * - Down boundary (inner wall/axis): only processes with is_down_boundary  
 * - Up boundary (outer wall): only processes with is_up_boundary
 * 
 * @param fields Physical field arrays
 * @param u Conservative variable arrays
 * @param grid Grid geometry
 * @param domain Domain decomposition info with boundary flags
 * @param params Simulation parameters
 * @param r_0 Reference radius for magnetic field
 */
void ApplyBoundaryConditions2D(PhysicalFields& fields, ConservativeVars& u, 
                                const GridGeometry& grid, const DomainInfo& domain,
                                const SimulationParams& params, double r_0);

// Legacy 1D version (kept for compatibility)
void ApplyBoundaryConditions(PhysicalFields& fields, ConservativeVars& u, 
                               const GridGeometry& grid, const DomainInfo& domain,
                               const SimulationParams& params, double r_0);
