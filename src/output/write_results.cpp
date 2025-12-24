#include "output/write_results.hpp"
#include <omp.h>

#include <iomanip>
#include <sstream>

#include "integrals.hpp"
#include "memory.hpp"
#include "mpi_comm.hpp"
#include "output/write_plt.hpp"
#include "output/write_vtk.hpp"

auto GenerateOutputFilename(const std::string& format, int frame_number, int mpi_size,
                            int omp_threads, const std::string& filename_template,
                            int L_max, int M_max, double dt) -> std::string {
    const std::string ext = (format == "vtk") ? "vtk" : (format == "plt") ? "plt" : "dat";

    std::ostringstream filename;

    if (filename_template == "parallel") {
        filename << "output__MPI_" << std::setw(3) << std::setfill('0') << mpi_size
                 << "__OMP_" << std::setw(3) << std::setfill('0') << omp_threads
                 << "__frame_" << std::setw(6) << std::setfill('0') << frame_number << '.'
                 << ext;
    } else {
        filename << "res_" << L_max << "x" << M_max << "__dt_" << std::fixed
                 << std::setprecision(7) << dt << "__frame_" << std::setw(6)
                 << std::setfill('0') << frame_number << '.' << ext;
    }

    return filename.str();
}

void WriteOutputFile(const char* filename, const SimulationParams& params,
                     const GridGeometry& grid, const PhysicalFields& fields) {
    if (params.output_format == "vtk") {
        WriteVtk(filename, params.L_max_global, params.M_max, params.dz,
                 grid.r, fields.rho, fields.v_z, fields.v_r, fields.v_phi,
                 fields.e, fields.H_z, fields.H_r, fields.H_phi, params.output_dir);
    } else if (params.output_format == "plt") {
        WritePlt(filename, params.L_max_global, params.M_max, params.dz,
                 grid.r, fields.rho, fields.v_z, fields.v_r, fields.v_phi,
                 fields.e, fields.H_z, fields.H_r, fields.H_phi, params.output_dir);
    }
}

void PrintDiagnostics(const PhysicalFields& fields, const GridGeometry& grid,
                      const SimulationParams& params) {
    const int idx = params.L_max_global;
    
    double mass_flux = GetMassFlux(fields.rho[idx], fields.v_z[idx],
                                   grid.dr[idx], params.M_max);
    double thrust = GetThrust(fields.rho[idx], fields.v_z[idx], fields.p[idx],
                              fields.H_r[idx], fields.H_phi[idx], fields.H_z[idx],
                              grid.dr[idx], params.M_max);
    
    printf("Mass flux: %.6f\n", mass_flux);
    printf("Thrust:    %.6f\n", thrust);
}

void WriteFinalOutput(const PhysicalFields& fields, const GridGeometry& grid,
                      const DomainInfo& domain, const SimulationParams& params,
                      PhysicalFields& global_fields_anim, GridGeometry& global_grid_anim,
                      int step_count, int frame_count, double t) {
    
    if (!params.animate) {
        // Non-animation mode: write single final output
        PhysicalFields global_fields;
        GridGeometry global_grid;

        if (domain.rank == 0) {
            AllocateFields(global_fields, params.L_max_global + 1, params.M_max + 1);
            MemoryAllocation2D(global_grid.r, params.L_max_global + 1, params.M_max + 1);
        }

        GatherResultsToRank0(fields, grid, domain, params, global_fields, global_grid);

        if (domain.rank == 0) {
            std::string filename = GenerateOutputFilename(
                params.output_format, frame_count, domain.size, omp_get_max_threads(),
                params.filename_template, params.L_max_global, params.M_max, params.dt);

            WriteOutputFile(filename.c_str(), params, global_grid, global_fields);
            printf("Final output written to: %s/%s\n", params.output_dir.c_str(),
                   filename.c_str());

            PrintDiagnostics(global_fields_anim, global_grid_anim, params);

            // Clean up
            DeallocateFields(global_fields, params.L_max_global + 1);
            MemoryClearing2D(global_grid.r, params.L_max_global + 1);
        }
    } else {
        // Animation mode: write final frame if needed
        bool need_final_frame = (step_count % params.animation_frequency != 0);
        
        if (need_final_frame) {
            GatherResultsToRank0(fields, grid, domain, params,
                               global_fields_anim, global_grid_anim);

            if (domain.rank == 0) {
                std::string filename = GenerateOutputFilename(
                    params.output_format, frame_count, domain.size, omp_get_max_threads(),
                    params.filename_template, params.L_max_global, params.M_max, params.dt);

                WriteOutputFile(filename.c_str(), params, global_grid_anim,
                              global_fields_anim);
                
                printf("Final frame %d written at step %d (t=%.6f): %s\n",
                       frame_count, step_count, t, filename.c_str());
            }
        }

        if (domain.rank == 0) {
            PrintDiagnostics(global_fields_anim, global_grid_anim, params);

            // Clean up animation arrays
            DeallocateFields(global_fields_anim, params.L_max_global + 1);
            MemoryClearing2D(global_grid_anim.r, params.L_max_global + 1);
        }
    }
}
