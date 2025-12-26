#pragma once

#include "types.hpp"
#include "decomposition.hpp"

void GatherResultsToRank0(const PhysicalFields& fields, const GridGeometry& grid,
                          const DomainInfo& domain, const SimulationParams& params,
                          PhysicalFields& global_fields, GridGeometry& global_grid);

void ExchangeGhostCellsConservative(ConservativeVars& u0, const DomainInfo& domain, int M_max);
void ExchangeGhostCellsPhysical(PhysicalFields& fields, const DomainInfo& domain, int M_max);
