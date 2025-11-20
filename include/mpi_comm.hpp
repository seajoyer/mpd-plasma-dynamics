#pragma once
#include "types.hpp"

// Exchange ghost cells for conservative variables
void ExchangeGhostCellsConservative(ConservativeVars& u0, const DomainInfo& domain, int M_max);

// Exchange ghost cells for physical variables
void ExchangeGhostCellsPhysical(PhysicalFields& fields, const DomainInfo& domain, int M_max);
