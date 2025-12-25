#pragma once
#include "types.hpp"
#include <mpi.h>

// ============================================================================
// 2D Domain Decomposition Setup
// ============================================================================

/**
 * @brief Initialize 2D Cartesian topology for domain decomposition
 * 
 * Creates a 2D Cartesian communicator and sets up neighbor relationships.
 * The decomposition prioritizes balance in the L direction first, then M.
 * 
 * @param domain Domain info structure to populate
 * @param params Simulation parameters with global grid dimensions
 */
void Setup2DDecomposition(DomainInfo& domain, const SimulationParams& params);

/**
 * @brief Compute optimal process grid dimensions
 * 
 * Finds dims[0] x dims[1] = size that best matches the aspect ratio
 * of the computational domain (L_max x M_max) for load balancing.
 * 
 * @param size Total number of MPI processes
 * @param L_max Global L dimension
 * @param M_max Global M dimension
 * @param dims Output array for dimensions [L_procs, M_procs]
 */
void ComputeOptimalDims(int size, int L_max, int M_max, int dims[2]);

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
 * @brief Get the Cartesian communicator as MPI_Comm type
 */
inline MPI_Comm GetCartComm(const DomainInfo& domain) {
    return static_cast<MPI_Comm>(domain.cart_comm);
}

/**
 * @brief Print domain decomposition info (for debugging)
 */
void PrintDecompositionInfo(const DomainInfo& domain, const SimulationParams& params);
