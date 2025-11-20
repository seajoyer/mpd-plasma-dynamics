#pragma once
#include "types.hpp"

// Exchange ghost cells for conservative variables
void exchange_ghost_cells_conservative(ConservativeVars& u0, const DomainInfo& domain, int M_max);

// Exchange ghost cells for physical variables
void exchange_ghost_cells_physical(PhysicalFields& fields, const DomainInfo& domain, int M_max);
