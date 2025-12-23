#pragma once

#include "types.hpp"

auto GenerateOutputFilename(const std::string& format, int frame_number, int mpi_size,
                            int omp_threads, const std::string& filename_template,
                            int L_max, int M_max, double dt) -> std::string;

void WriteOutputFile(const char* filename, const SimulationParams& params,
                     const GridGeometry& grid, const PhysicalFields& fields);

void PrintDiagnostics(const PhysicalFields& fields, const GridGeometry& grid,
                      const SimulationParams& params);

void WriteFinalOutput(const PhysicalFields& fields, const GridGeometry& grid,
                      const DomainInfo& domain, const SimulationParams& params,
                      PhysicalFields& global_fields_anim, GridGeometry& global_grid_anim,
                      int step_count, int frame_count, double t);
