#pragma once

#include "types.hpp"
#include <mpi.h>

void SetupCartesianTopology(DomainInfo& domain, int total_procs);

void SetupDomainDecomposition(DomainInfo& domain, SimulationParams& params);

void PrintDomainInfo(const DomainInfo& domain);
