#pragma once

#include "types.hpp"

void MemoryAllocation2D(double**& array, int rows, int columns);
void MemoryClearing2D(double** &array, int rows);

void AllocateFields(PhysicalFields& fields, int rows, int cols);
void DeallocateFields(PhysicalFields& fields, int rows);
void AllocateConservative(ConservativeVars& u, int rows, int cols);
void DeallocateConservative(ConservativeVars& u, int rows);
void AllocatePreviousState(PreviousState& prev, int rows, int cols);
void DeallocatePreviousState(PreviousState& prev, int rows);
