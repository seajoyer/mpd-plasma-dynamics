#pragma once
#include "types.hpp"

/**
 * @brief Compute optimal process grid dimensions for 2D decomposition
 * 
 * Finds dims[0] x dims[1] = size that best matches the aspect ratio
 * of the computational domain (L_max x M_max) for load balancing.
 */
void ComputeOptimalDims(int size, int L_max, int M_max, int dims[2]);

/**
 * @brief Set up MPI Cartesian topology for 2D domain decomposition
 * 
 * Creates a 2D Cartesian communicator, determines neighbor ranks,
 * and sets boundary flags.
 */
void SetupCartesianTopology(DomainInfo& domain, int L_max, int M_max);

/**
 * @brief Compute local domain extents for L and M directions
 * 
 * Distributes cells evenly among processes, handling remainder.
 */
void ComputeLocalDomainExtents(DomainInfo& domain, int L_max_global, int M_max);

/**
 * @brief Full 2D decomposition setup (combines topology + extents)
 */
void Setup2DDecomposition(DomainInfo& domain, const SimulationParams& params);

/**
 * @brief Print domain decomposition info for debugging
 */
void PrintDecompositionInfo(const DomainInfo& domain, const SimulationParams& params);
