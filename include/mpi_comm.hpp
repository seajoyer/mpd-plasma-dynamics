#pragma once
#include "types.hpp"
#include "domain_decomposition.hpp"
#include <mpi.h>

// ============================================================================
// Ghost Cell Exchange - 2D Version
// ============================================================================

/**
 * @brief Exchange ghost cells for conservative variables in 2D
 * 
 * Exchanges ghost cells with all 4 neighbors (left, right, up, down).
 * Uses non-blocking communication for efficiency.
 * 
 * @param u0 Conservative variables to exchange
 * @param domain Domain decomposition info
 * @param params Simulation parameters
 */
void ExchangeGhostCellsConservative2D(ConservativeVars& u0, const DomainInfo& domain,
                                       const SimulationParams& params);

/**
 * @brief Exchange ghost cells for physical variables in 2D
 * 
 * @param fields Physical fields to exchange
 * @param domain Domain decomposition info  
 * @param params Simulation parameters
 */
void ExchangeGhostCellsPhysical2D(PhysicalFields& fields, const DomainInfo& domain,
                                   const SimulationParams& params);

// ============================================================================
// Data Gathering for Output - 2D Version
// ============================================================================

/**
 * @brief Gather all results to rank 0 from 2D decomposed domain
 * 
 * Collects data from all processes and assembles into global arrays on rank 0.
 * Uses a two-stage gather: first gather columns within each row of processes,
 * then gather rows to rank 0.
 * 
 * @param fields Local physical fields
 * @param grid Local grid geometry
 * @param domain Domain decomposition info
 * @param params Simulation parameters
 * @param global_fields Output: global physical fields (only valid on rank 0)
 * @param global_grid Output: global grid geometry (only valid on rank 0)
 */
void GatherResultsToRank0_2D(const PhysicalFields& fields, const GridGeometry& grid,
                              const DomainInfo& domain, const SimulationParams& params,
                              PhysicalFields& global_fields, GridGeometry& global_grid);

// ============================================================================
// Legacy 1D functions (kept for compatibility during transition)
// ============================================================================

void GatherResultsToRank0(const PhysicalFields& fields, const GridGeometry& grid,
                          const DomainInfo& domain, const SimulationParams& params,
                          PhysicalFields& global_fields, GridGeometry& global_grid);

void ExchangeGhostCellsConservative(ConservativeVars& u0, const DomainInfo& domain, int M_max);

void ExchangeGhostCellsPhysical(PhysicalFields& fields, const DomainInfo& domain, int M_max);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get the Cartesian communicator
 */
inline MPI_Comm GetCartComm(const DomainInfo& domain) {
    return domain.cart_comm;
}
