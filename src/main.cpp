#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

#include "boundary.hpp"
#include "geometry.hpp"
#include "memory.hpp"
#include "mpi_comm.hpp"
#include "output/write_plt.hpp"
#include "output/write_vtk.hpp"
#include "physics.hpp"
#include "solver.hpp"
#include "types.hpp"

// Helper function to generate numbered output filenames
auto GenerateOutputFilename(const std::string& format, int frame_number) -> std::string {
    const std::string ext = (format == "vtk") ? "vtk" : (format == "plt") ? "plt" : "dat";

    std::ostringstream filename;
    filename << "output_MHD_" << std::setw(4) << std::setfill('0') << frame_number << '.'
             << ext;
    return filename.str();
}

// Helper function to allocate all field arrays
void AllocateFields(PhysicalFields& fields, int rows, int cols) {
    MemoryAllocation2D(fields.rho, rows, cols);
    MemoryAllocation2D(fields.v_r, rows, cols);
    MemoryAllocation2D(fields.v_phi, rows, cols);
    MemoryAllocation2D(fields.v_z, rows, cols);
    MemoryAllocation2D(fields.e, rows, cols);
    MemoryAllocation2D(fields.p, rows, cols);
    MemoryAllocation2D(fields.P, rows, cols);
    MemoryAllocation2D(fields.H_r, rows, cols);
    MemoryAllocation2D(fields.H_phi, rows, cols);
    MemoryAllocation2D(fields.H_z, rows, cols);
}

void DeallocateFields(PhysicalFields& fields, int rows) {
    MemoryClearing2D(fields.rho, rows);
    MemoryClearing2D(fields.v_r, rows);
    MemoryClearing2D(fields.v_phi, rows);
    MemoryClearing2D(fields.v_z, rows);
    MemoryClearing2D(fields.e, rows);
    MemoryClearing2D(fields.p, rows);
    MemoryClearing2D(fields.P, rows);
    MemoryClearing2D(fields.H_r, rows);
    MemoryClearing2D(fields.H_phi, rows);
    MemoryClearing2D(fields.H_z, rows);
}

void AllocateConservative(ConservativeVars& u, int rows, int cols) {
    MemoryAllocation2D(u.u_1, rows, cols);
    MemoryAllocation2D(u.u_2, rows, cols);
    MemoryAllocation2D(u.u_3, rows, cols);
    MemoryAllocation2D(u.u_4, rows, cols);
    MemoryAllocation2D(u.u_5, rows, cols);
    MemoryAllocation2D(u.u_6, rows, cols);
    MemoryAllocation2D(u.u_7, rows, cols);
    MemoryAllocation2D(u.u_8, rows, cols);
}

void DeallocateConservative(ConservativeVars& u, int rows) {
    MemoryClearing2D(u.u_1, rows);
    MemoryClearing2D(u.u_2, rows);
    MemoryClearing2D(u.u_3, rows);
    MemoryClearing2D(u.u_4, rows);
    MemoryClearing2D(u.u_5, rows);
    MemoryClearing2D(u.u_6, rows);
    MemoryClearing2D(u.u_7, rows);
    MemoryClearing2D(u.u_8, rows);
}

void AllocatePreviousState(PreviousState& prev, int rows, int cols) {
    MemoryAllocation2D(prev.rho_prev, rows, cols);
    MemoryAllocation2D(prev.v_z_prev, rows, cols);
    MemoryAllocation2D(prev.v_r_prev, rows, cols);
    MemoryAllocation2D(prev.v_phi_prev, rows, cols);
    MemoryAllocation2D(prev.H_z_prev, rows, cols);
    MemoryAllocation2D(prev.H_r_prev, rows, cols);
    MemoryAllocation2D(prev.H_phi_prev, rows, cols);
}

void DeallocatePreviousState(PreviousState& prev, int rows) {
    MemoryClearing2D(prev.rho_prev, rows);
    MemoryClearing2D(prev.v_z_prev, rows);
    MemoryClearing2D(prev.v_r_prev, rows);
    MemoryClearing2D(prev.v_phi_prev, rows);
    MemoryClearing2D(prev.H_z_prev, rows);
    MemoryClearing2D(prev.H_r_prev, rows);
    MemoryClearing2D(prev.H_phi_prev, rows);
}

void GatherResultsToRank0(const PhysicalFields& fields, const GridGeometry& grid,
                          const DomainInfo& domain, const SimulationParams& params,
                          PhysicalFields& global_fields, GridGeometry& global_grid) {
    const int M_max = params.M_max;
    const int L_max_global = params.L_max_global;
    const int local_L = domain.local_L;
    const int L_per_proc = domain.L_per_proc;

    // Gather data row by row (excluding ghost cells)
    for (int m = 0; m < M_max + 1; m++) {
        auto* local_row_rho = new double[local_L];
        auto* local_row_vz = new double[local_L];
        auto* local_row_vr = new double[local_L];
        auto* local_row_vphi = new double[local_L];
        auto* local_row_e = new double[local_L];
        auto* local_row_Hz = new double[local_L];
        auto* local_row_Hr = new double[local_L];
        auto* local_row_Hphi = new double[local_L];
        auto* local_row_r = new double[local_L];

        for (int l = 0; l < local_L; l++) {
            local_row_rho[l] = fields.rho[l + 1][m];
            local_row_vz[l] = fields.v_z[l + 1][m];
            local_row_vr[l] = fields.v_r[l + 1][m];
            local_row_vphi[l] = fields.v_phi[l + 1][m];
            local_row_e[l] = fields.e[l + 1][m];
            local_row_Hz[l] = fields.H_z[l + 1][m];
            local_row_Hr[l] = fields.H_r[l + 1][m];
            local_row_Hphi[l] = fields.H_phi[l + 1][m];
            local_row_r[l] = grid.r[l + 1][m];
        }

        double *global_row_rho = nullptr, *global_row_vz = nullptr,
               *global_row_vr = nullptr;
        double *global_row_vphi = nullptr, *global_row_e = nullptr;
        double *global_row_Hz = nullptr, *global_row_Hr = nullptr,
               *global_row_Hphi = nullptr;
        double* global_row_r = nullptr;

        if (domain.rank == 0) {
            global_row_rho = new double[L_max_global];
            global_row_vz = new double[L_max_global];
            global_row_vr = new double[L_max_global];
            global_row_vphi = new double[L_max_global];
            global_row_e = new double[L_max_global];
            global_row_Hz = new double[L_max_global];
            global_row_Hr = new double[L_max_global];
            global_row_Hphi = new double[L_max_global];
            global_row_r = new double[L_max_global];
        }

        int* recvcounts = new int[domain.size];
        int* displs = new int[domain.size];

        for (int i = 0; i < domain.size; i++) {
            int i_start = i * L_per_proc;
            int i_end = (i + 1) * L_per_proc - 1;
            if (i == domain.size - 1) i_end = L_max_global - 1;
            recvcounts[i] = i_end - i_start + 1;
            displs[i] = i_start;
        }

        MPI_Gatherv(local_row_rho, local_L, MPI_DOUBLE, global_row_rho, recvcounts,
                    displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_vz, local_L, MPI_DOUBLE, global_row_vz, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_vr, local_L, MPI_DOUBLE, global_row_vr, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_vphi, local_L, MPI_DOUBLE, global_row_vphi, recvcounts,
                    displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_e, local_L, MPI_DOUBLE, global_row_e, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_Hz, local_L, MPI_DOUBLE, global_row_Hz, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_Hr, local_L, MPI_DOUBLE, global_row_Hr, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_Hphi, local_L, MPI_DOUBLE, global_row_Hphi, recvcounts,
                    displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_r, local_L, MPI_DOUBLE, global_row_r, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (domain.rank == 0) {
            for (int l = 0; l < L_max_global; l++) {
                global_fields.rho[l][m] = global_row_rho[l];
                global_fields.v_z[l][m] = global_row_vz[l];
                global_fields.v_r[l][m] = global_row_vr[l];
                global_fields.v_phi[l][m] = global_row_vphi[l];
                global_fields.e[l][m] = global_row_e[l];
                global_fields.H_z[l][m] = global_row_Hz[l];
                global_fields.H_r[l][m] = global_row_Hr[l];
                global_fields.H_phi[l][m] = global_row_Hphi[l];
                global_grid.r[l][m] = global_row_r[l];
            }

            delete[] global_row_rho;
            delete[] global_row_vz;
            delete[] global_row_vr;
            delete[] global_row_vphi;
            delete[] global_row_e;
            delete[] global_row_Hz;
            delete[] global_row_Hr;
            delete[] global_row_Hphi;
            delete[] global_row_r;
        }

        delete[] local_row_rho;
        delete[] local_row_vz;
        delete[] local_row_vr;
        delete[] local_row_vphi;
        delete[] local_row_e;
        delete[] local_row_Hz;
        delete[] local_row_Hr;
        delete[] local_row_Hphi;
        delete[] local_row_r;
        delete[] recvcounts;
        delete[] displs;
    }
}

auto main(int argc, char* argv[]) -> int {
    // Initialize MPI
    DomainInfo domain;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &domain.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &domain.size);

    // Simulation parameters
    SimulationParams params;
    params.gamma = 1.67;
    params.beta = 0.05;
    params.H_z0 = 0.25;
    params.animate = 0;
    params.animation_frequency = 50000;
    params.output_format = "vtk";
    params.output_dir = "output";

    params.convergence_threshold = 1e-5;
    params.check_frequency = 500;

    params.T = 50.0;
    params.dt = 0.0000125;

    params.L_max_global = 800;
    params.L_end = 265;
    params.M_max = 400;

    params.dz = 1.0 / params.L_max_global;
    params.dy = 1.0 / params.M_max;

    // Domain decomposition
    domain.L_per_proc = params.L_max_global / domain.size;
    domain.l_start = domain.rank * domain.L_per_proc;
    domain.l_end = (domain.rank + 1) * domain.L_per_proc - 1;
    if (domain.rank == domain.size - 1) {
        domain.l_end = params.L_max_global - 1;
    }
    domain.local_L = domain.l_end - domain.l_start + 1;
    domain.local_L_with_ghosts = domain.local_L + 2;

    // Parse command line arguments
    int procs = 1;
    if (argc > 1) {
        procs = atoi(argv[1]);
    }

    for (int i = 2; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--converge") {
            params.convergence_threshold = atof(argv[i + 1]);
            if (domain.rank == 0) {
                printf("Convergence checking enabled: threshold = %e\n",
                       params.convergence_threshold);
            }
            i++;
        } else if (std::string(argv[i]) == "--animate") {
            params.animate = 1;
            if (domain.rank == 0) {
                printf("Animation output enabled\n");
            }
        } else if (std::string(argv[i]) == "--anim-freq") {
            params.animation_frequency = atoi(argv[i + 1]);
            if (domain.rank == 0) {
                printf("Animation frequency: every %d steps\n",
                       params.animation_frequency);
            }
            i++;
        } else if (std::string(argv[i]) == "--check-freq") {
            params.check_frequency = atoi(argv[i + 1]);
            if (domain.rank == 0) {
                printf("Check frequency: every %d steps\n",
                       params.check_frequency);
            }
            i++;
        } else if (std::string(argv[i]) == "--format") {
            params.output_format = std::string(argv[i + 1]);
            if (domain.rank == 0) {
                printf("Output format: %s\n", params.output_format.c_str());
            }
            i++;
        } else if (std::string(argv[i]) == "--output-dir") {
            params.output_dir = std::string(argv[i + 1]);
            if (domain.rank == 0) {
                printf("Output directory: %s\n", params.output_dir.c_str());
            }
            i++;
        }
    }

    omp_set_num_threads(procs);

    double begin, end, total;

    // Allocate arrays
    PhysicalFields fields;
    ConservativeVars u, u0;
    GridGeometry grid;

    AllocateFields(fields, domain.local_L_with_ghosts, params.M_max + 1);
    AllocateConservative(u, domain.local_L_with_ghosts, params.M_max + 1);
    AllocateConservative(u0, domain.local_L_with_ghosts, params.M_max + 1);

    MemoryAllocation2D(grid.r, domain.local_L_with_ghosts, params.M_max + 1);
    MemoryAllocation2D(grid.r_z, domain.local_L_with_ghosts, params.M_max + 1);
    grid.R = new double[domain.local_L_with_ghosts];
    grid.dr = new double[domain.local_L_with_ghosts];

    for (int l = 0; l < domain.local_L_with_ghosts; l++) {
        grid.R[l] = 0;
        grid.dr[l] = 0;
    }

    PreviousState prev_state;
    if (params.convergence_threshold > 0) {
        AllocatePreviousState(prev_state, domain.local_L_with_ghosts, params.M_max + 1);
    }

    // Initialize grid geometry
    double r_0 = (R1(0) + R2(0)) / 2.0;

    for (int l = 0; l < domain.local_L_with_ghosts; l++) {
        int l_global = domain.l_start + l - 1;
        double z = l_global * params.dz;

        grid.R[l] = R2(z) - R1(z);
        grid.dr[l] = grid.R[l] / params.M_max;

        for (int m = 0; m < params.M_max + 1; m++) {
            grid.r[l][m] = (1 - m * params.dy) * R1(z) + m * params.dy * R2(z);
            grid.r_z[l][m] = (1 - m * params.dy) * DerR1(z) + m * params.dy * DerR2(z);
        }
    }

// Initialize physical fields
#pragma omp parallel for collapse(2)
    for (int l = 1; l < domain.local_L_with_ghosts - 1; l++) {
        for (int m = 0; m < params.M_max + 1; m++) {
            int l_global = domain.l_start + l - 1;

            fields.rho[l][m] = 1.0;
            fields.v_z[l][m] = 0.1;
            fields.v_r[l][m] = 0.1;
            fields.v_phi[l][m] = 0;
            fields.H_phi[l][m] = (1 - 0.9 * l_global * params.dz) * r_0 / grid.r[l][m];
            fields.H_z[l][m] = params.H_z0;
            fields.H_r[l][m] = fields.H_z[l][m] * grid.r_z[l][m];

            fields.e[l][m] = params.beta / (2.0 * (params.gamma - 1.0));
            fields.p[l][m] = params.beta / 2.0;
            fields.P[l][m] = fields.p[l][m] +
                             0.5 * (pow(fields.H_z[l][m], 2) + pow(fields.H_r[l][m], 2) +
                                    pow(fields.H_phi[l][m], 2));
        }
    }

    // Initialize previous state if convergence checking is enabled
    if (params.convergence_threshold > 0) {
#pragma omp parallel for collapse(2)
        for (int l = 0; l < domain.local_L_with_ghosts; l++) {
            for (int m = 0; m < params.M_max + 1; m++) {
                prev_state.rho_prev[l][m] = fields.rho[l][m];
                prev_state.v_z_prev[l][m] = fields.v_z[l][m];
                prev_state.v_r_prev[l][m] = fields.v_r[l][m];
                prev_state.v_phi_prev[l][m] = fields.v_phi[l][m];
                prev_state.H_z_prev[l][m] = fields.H_z[l][m];
                prev_state.H_r_prev[l][m] = fields.H_r[l][m];
                prev_state.H_phi_prev[l][m] = fields.H_phi[l][m];
            }
        }
    }

    // Initialize conservative variables
    InitializeConservativeVars(u0, fields, grid, domain.local_L_with_ghosts,
                                 params.M_max);

    // Initialize ghost cells for boundary ranks before first time step
    if (domain.rank == 0) {
// Copy first physical cell to left ghost cell
#pragma omp parallel for
        for (int m = 0; m < params.M_max + 1; m++) {
            fields.rho[0][m] = fields.rho[1][m];
            fields.v_z[0][m] = fields.v_z[1][m];
            fields.v_r[0][m] = fields.v_r[1][m];
            fields.v_phi[0][m] = fields.v_phi[1][m];
            fields.H_phi[0][m] = fields.H_phi[1][m];
            fields.H_z[0][m] = fields.H_z[1][m];
            fields.H_r[0][m] = fields.H_r[1][m];
            fields.e[0][m] = fields.e[1][m];
            fields.p[0][m] = fields.p[1][m];
            fields.P[0][m] = fields.P[1][m];

            u0.u_1[0][m] = fields.rho[0][m] * grid.r[0][m];
            u0.u_2[0][m] = fields.rho[0][m] * fields.v_z[0][m] * grid.r[0][m];
            u0.u_3[0][m] = fields.rho[0][m] * fields.v_r[0][m] * grid.r[0][m];
            u0.u_4[0][m] = fields.rho[0][m] * fields.v_phi[0][m] * grid.r[0][m];
            u0.u_5[0][m] = fields.rho[0][m] * fields.e[0][m] * grid.r[0][m];
            u0.u_6[0][m] = fields.H_phi[0][m];
            u0.u_7[0][m] = fields.H_z[0][m] * grid.r[0][m];
            u0.u_8[0][m] = fields.H_r[0][m] * grid.r[0][m];
        }
    }

    if (domain.rank == domain.size - 1) {
// Copy last physical cell to right ghost cell
#pragma omp parallel for
        for (int m = 0; m < params.M_max + 1; m++) {
            fields.rho[domain.local_L + 1][m] = fields.rho[domain.local_L][m];
            fields.v_z[domain.local_L + 1][m] = fields.v_z[domain.local_L][m];
            fields.v_r[domain.local_L + 1][m] = fields.v_r[domain.local_L][m];
            fields.v_phi[domain.local_L + 1][m] = fields.v_phi[domain.local_L][m];
            fields.H_phi[domain.local_L + 1][m] = fields.H_phi[domain.local_L][m];
            fields.H_z[domain.local_L + 1][m] = fields.H_z[domain.local_L][m];
            fields.H_r[domain.local_L + 1][m] = fields.H_r[domain.local_L][m];
            fields.e[domain.local_L + 1][m] = fields.e[domain.local_L][m];
            fields.p[domain.local_L + 1][m] = fields.p[domain.local_L][m];
            fields.P[domain.local_L + 1][m] = fields.P[domain.local_L][m];

            u0.u_1[domain.local_L + 1][m] =
                fields.rho[domain.local_L + 1][m] * grid.r[domain.local_L + 1][m];
            u0.u_2[domain.local_L + 1][m] = fields.rho[domain.local_L + 1][m] *
                                            fields.v_z[domain.local_L + 1][m] *
                                            grid.r[domain.local_L + 1][m];
            u0.u_3[domain.local_L + 1][m] = fields.rho[domain.local_L + 1][m] *
                                            fields.v_r[domain.local_L + 1][m] *
                                            grid.r[domain.local_L + 1][m];
            u0.u_4[domain.local_L + 1][m] = fields.rho[domain.local_L + 1][m] *
                                            fields.v_phi[domain.local_L + 1][m] *
                                            grid.r[domain.local_L + 1][m];
            u0.u_5[domain.local_L + 1][m] = fields.rho[domain.local_L + 1][m] *
                                            fields.e[domain.local_L + 1][m] *
                                            grid.r[domain.local_L + 1][m];
            u0.u_6[domain.local_L + 1][m] = fields.H_phi[domain.local_L + 1][m];
            u0.u_7[domain.local_L + 1][m] =
                fields.H_z[domain.local_L + 1][m] * grid.r[domain.local_L + 1][m];
            u0.u_8[domain.local_L + 1][m] =
                fields.H_r[domain.local_L + 1][m] * grid.r[domain.local_L + 1][m];
        }
    }

    // Start timing
    if (domain.rank == 0) {
        begin = MPI_Wtime();
    }

    double t = 0.0;
    int step_count = 0;
    int frame_count = 0;
    bool converged = false;

    // Allocate global arrays for animation output (rank 0 only)
    PhysicalFields global_fields_anim;
    GridGeometry global_grid_anim;

    if (domain.rank == 0 && params.animate) {
        AllocateFields(global_fields_anim, params.L_max_global + 1, params.M_max + 1);
        MemoryAllocation2D(global_grid_anim.r, params.L_max_global + 1,
                             params.M_max + 1);

        printf("Animation enabled: will output every %d steps\n",
               params.animation_frequency);
        printf("Output format: %s\n", params.output_format.c_str());
    }

    // Output initial conditions as frame 0 if animation is enabled
    if (params.animate) {
        GatherResultsToRank0(fields, grid, domain, params, global_fields_anim,
                             global_grid_anim);

        if (domain.rank == 0) {
            std::string filename =
                GenerateOutputFilename(params.output_format, frame_count);

            if (params.output_format == "vtk") {
                WriteVtk(filename.c_str(), params.L_max_global, params.M_max,
                         params.dz, global_grid_anim.r, global_fields_anim.rho,
                         global_fields_anim.v_z, global_fields_anim.v_r,
                         global_fields_anim.v_phi, global_fields_anim.e,
                         global_fields_anim.H_z, global_fields_anim.H_r,
                         global_fields_anim.H_phi, params.output_dir);
            } else if (params.output_format == "plt") {
                WritePlt(filename.c_str(), params.L_max_global, params.M_max,
                         params.dz, global_grid_anim.r, global_fields_anim.rho,
                         global_fields_anim.v_z, global_fields_anim.v_r,
                         global_fields_anim.v_phi, global_fields_anim.e,
                         global_fields_anim.H_z, global_fields_anim.H_r,
                         global_fields_anim.H_phi, params.output_dir);
            }

            printf("Frame %d written (initial conditions, t=%.6f): %s\n", frame_count,
                   t, filename.c_str());
            frame_count++;
        }
    }

    // Main time loop
    while (t < params.T && !converged) {
        // Exchange ghost cells
        ExchangeGhostCellsConservative(u0, domain, params.M_max);
        ExchangeGhostCellsPhysical(fields, domain, params.M_max);

        // Compute one time step
        ComputeTimeStep(u, u0, fields, grid, domain, params);

        // Update central part
        UpdatePhysicalFields(fields, u, grid, domain.local_L + 2, params.M_max,
                               params.gamma);

        // Apply boundary conditions
        ApplyBoundaryConditions(fields, u, grid, domain, params, r_0);

        // Data update
        UpdatePhysicalFields(fields, u, grid, domain.local_L_with_ghosts, params.M_max,
                               params.gamma);

        // Copy u to u0
        CopyConservativeVars(u0, u, domain.local_L_with_ghosts, params.M_max);

        // Convergence check
        if (params.convergence_threshold > 0 &&
            step_count % params.check_frequency == 0) {
            double change = ComputeSolutionChange(
                fields.rho, prev_state.rho_prev, fields.v_z, prev_state.v_z_prev,
                fields.v_r, prev_state.v_r_prev, fields.v_phi, prev_state.v_phi_prev,
                fields.H_z, prev_state.H_z_prev, fields.H_r, prev_state.H_r_prev,
                fields.H_phi, prev_state.H_phi_prev, domain.local_L, params.M_max);

            if (domain.rank == 0) {
                printf("Step %d, t=%.6f, relative change: %.6e\n", step_count, t, change);
            }

            if (change < params.convergence_threshold) {
                converged = true;
                if (domain.rank == 0) {
                    printf("Converged at t=%.6f after %d steps\n", t, step_count);
                }
            }

// Update previous state
#pragma omp parallel for collapse(2)
            for (int l = 1; l < domain.local_L + 1; l++) {
                for (int m = 0; m < params.M_max + 1; m++) {
                    prev_state.rho_prev[l][m] = fields.rho[l][m];
                    prev_state.v_z_prev[l][m] = fields.v_z[l][m];
                    prev_state.v_r_prev[l][m] = fields.v_r[l][m];
                    prev_state.v_phi_prev[l][m] = fields.v_phi[l][m];
                    prev_state.H_z_prev[l][m] = fields.H_z[l][m];
                    prev_state.H_r_prev[l][m] = fields.H_r[l][m];
                    prev_state.H_phi_prev[l][m] = fields.H_phi[l][m];
                }
            }
        }

        // CFL stability check (every 100 steps)
        if (step_count % 100 == 0) {
            double max_wave_speed = ComputeMaxWaveSpeed(
                fields.rho, fields.v_z, fields.v_r, fields.v_phi, fields.H_z, fields.H_r,
                fields.H_phi, fields.p, domain.local_L, params.M_max, params.gamma);
            double dx = std::min(params.dz, params.dy);
            double CFL_number = 0.5;
            double dt_max = CFL_number * dx / (max_wave_speed + 1e-10);

            if (params.dt > dt_max && domain.rank == 0 && step_count % 1000 == 0) {
                printf(
                    "WARNING: dt=%.6e exceeds CFL limit dt_max=%.6e (max_speed=%.3f)\n",
                    params.dt, dt_max, max_wave_speed);
            }
        }

        t += params.dt;
        step_count++;

        // Debug output
        if (domain.rank == 0 && step_count % 1000 == 0) {
            int check_l = 20;
            int check_m = 40;
            if (check_l >= domain.l_start && check_l <= domain.l_end) {
                int local_check_l = check_l - domain.l_start + 1;
                printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", t,
                       fields.rho[local_check_l][check_m],
                       fields.v_z[local_check_l][check_m],
                       fields.v_phi[local_check_l][check_m],
                       fields.e[local_check_l][check_m],
                       fields.H_phi[local_check_l][check_m]);
            }
        }

        // Animation output
        if (params.animate && step_count % params.animation_frequency == 0) {
            GatherResultsToRank0(fields, grid, domain, params, global_fields_anim,
                                 global_grid_anim);

            if (domain.rank == 0) {
                std::string filename =
                    GenerateOutputFilename(params.output_format, frame_count);

                if (params.output_format == "vtk") {
                    WriteVtk(filename.c_str(), params.L_max_global, params.M_max,
                             params.dz, global_grid_anim.r, global_fields_anim.rho,
                             global_fields_anim.v_z, global_fields_anim.v_r,
                             global_fields_anim.v_phi, global_fields_anim.e,
                             global_fields_anim.H_z, global_fields_anim.H_r,
                             global_fields_anim.H_phi, params.output_dir);
                } else if (params.output_format == "plt") {
                    WritePlt(filename.c_str(), params.L_max_global, params.M_max,
                             params.dz, global_grid_anim.r, global_fields_anim.rho,
                             global_fields_anim.v_z, global_fields_anim.v_r,
                             global_fields_anim.v_phi, global_fields_anim.e,
                             global_fields_anim.H_z, global_fields_anim.H_r,
                             global_fields_anim.H_phi, params.output_dir);
                }

                printf("Frame %d written at step %d (t=%.6f): %s\n", frame_count,
                       step_count, t, filename.c_str());
                frame_count++;
            }
        }
    }

    // Finish timing
    if (domain.rank == 0) {
        end = MPI_Wtime();
        total = end - begin;
        printf("Calculation time : %lf sec\n", total);
        if (params.animate) {
            printf("Total animation frames generated: %d\n", frame_count);
        }
    }

    // Write final output file (always written, regardless of animation mode)
    if (!params.animate) {
        // Only allocate global arrays if not in animation mode (otherwise already
        // allocated)
        PhysicalFields global_fields;
        GridGeometry global_grid;

        if (domain.rank == 0) {
            AllocateFields(global_fields, params.L_max_global + 1, params.M_max + 1);
            MemoryAllocation2D(global_grid.r, params.L_max_global + 1,
                                 params.M_max + 1);
        }

        GatherResultsToRank0(fields, grid, domain, params, global_fields, global_grid);

        if (domain.rank == 0) {
            // VTK output
            if (params.output_format == "vtk") {
                WriteVtk("output_MHD.vtk", params.L_max_global, params.M_max, params.dz,
                         global_grid.r, global_fields.rho, global_fields.v_z,
                         global_fields.v_r, global_fields.v_phi, global_fields.e,
                         global_fields.H_z, global_fields.H_r, global_fields.H_phi,
                         params.output_dir);
                printf("Final output written to: %s/output_MHD.vtk\n", params.output_dir.c_str());
            }

            // tecplot output
            if (params.output_format == "plt") {
                WritePlt("output_MHD.plt", params.L_max_global, params.M_max, params.dz,
                         global_grid.r, global_fields.rho, global_fields.v_z,
                         global_fields.v_r, global_fields.v_phi, global_fields.e,
                         global_fields.H_z, global_fields.H_r, global_fields.H_phi,
                         params.output_dir);
                printf("Final output written to: %s/output_MHD.plt\n", params.output_dir.c_str());
            }

            // Clean up global arrays
            DeallocateFields(global_fields, params.L_max_global + 1);
            MemoryClearing2D(global_grid.r, params.L_max_global + 1);
        }
    } else {
        // In animation mode, write one final frame at the end if not already written
        if (step_count % params.animation_frequency != 0) {
            GatherResultsToRank0(fields, grid, domain, params, global_fields_anim,
                                 global_grid_anim);

            if (domain.rank == 0) {
                std::string filename =
                    GenerateOutputFilename(params.output_format, frame_count);

                if (params.output_format == "vtk") {
                    WriteVtk(filename.c_str(), params.L_max_global, params.M_max,
                             params.dz, global_grid_anim.r, global_fields_anim.rho,
                             global_fields_anim.v_z, global_fields_anim.v_r,
                             global_fields_anim.v_phi, global_fields_anim.e,
                             global_fields_anim.H_z, global_fields_anim.H_r,
                             global_fields_anim.H_phi, params.output_dir);
                } else if (params.output_format == "plt") {
                    WritePlt(filename.c_str(), params.L_max_global, params.M_max,
                             params.dz, global_grid_anim.r, global_fields_anim.rho,
                             global_fields_anim.v_z, global_fields_anim.v_r,
                             global_fields_anim.v_phi, global_fields_anim.e,
                             global_fields_anim.H_z, global_fields_anim.H_r,
                             global_fields_anim.H_phi, params.output_dir);
                }

                printf("Final frame %d written at step %d (t=%.6f): %s\n", frame_count,
                       step_count, t, filename.c_str());
            }
        }

        // Clean up animation global arrays
        if (domain.rank == 0) {
            DeallocateFields(global_fields_anim, params.L_max_global + 1);
            MemoryClearing2D(global_grid_anim.r, params.L_max_global + 1);
        }
    }

    // Clean up local arrays
    DeallocateFields(fields, domain.local_L_with_ghosts);
    DeallocateConservative(u, domain.local_L_with_ghosts);
    DeallocateConservative(u0, domain.local_L_with_ghosts);

    MemoryClearing2D(grid.r, domain.local_L_with_ghosts);
    MemoryClearing2D(grid.r_z, domain.local_L_with_ghosts);
    delete[] grid.R;
    delete[] grid.dr;

    if (params.convergence_threshold > 0) {
        DeallocatePreviousState(prev_state, domain.local_L_with_ghosts);
    }

    MPI_Finalize();
    return 0;
}
