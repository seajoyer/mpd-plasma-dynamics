#pragma once
#include "types.hpp"

void GatherResultsToRank0(const PhysicalFields& fields, const GridGeometry& grid,
                          const DomainInfo& domain, const SimulationParams& params,
                          PhysicalFields& global_fields, GridGeometry& global_grid);

// Exchange ghost cells for conservative variables
void ExchangeGhostCellsConservative(ConservativeVars& u0, const DomainInfo& domain, int M_max);

// Exchange ghost cells for physical variables
void ExchangeGhostCellsPhysical(PhysicalFields& fields, const DomainInfo& domain, int M_max);
