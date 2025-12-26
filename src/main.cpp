#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <string>

#include "boundary.hpp"
#include "geometry.hpp"
#include "memory.hpp"
#include "mpi_comm.hpp"
#include "output/write_plt.hpp"
#include "output/write_results.hpp"
#include "output/write_vtk.hpp"
#include "physics.hpp"
#include "solver.hpp"
#include "types.hpp"
#include "integrals.hpp"

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
    params.check_frequency = 100;

    params.T = 4.0;
    params.dt = 0.0000115;

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
        } else if (std::string(argv[i]) == "--filename-template") {
            params.filename_template = std::string(argv[i + 1]);
            if (domain.rank == 0) {
                printf("Filename template: %s\n", params.filename_template.c_str());
            }
            i++;
        } else if (std::string(argv[i]) == "--end-time") {
            params.T = atof(argv[i + 1]);
            if (domain.rank == 0) {
                printf("Maximum value for time: %e\n", params.T);
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
            std::string filename = GenerateOutputFilename(params.output_format, frame_count,
                                                        domain.size, omp_get_max_threads(),
                                                        params.filename_template,
                                                        params.L_max_global, params.M_max, params.dt);

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
                std::string filename = GenerateOutputFilename(params.output_format, frame_count,
                                                            domain.size, omp_get_max_threads(),
                                                            params.filename_template,
                                                            params.L_max_global, params.M_max, params.dt);

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
	printf("Mass flux : %f \n", GetMassFlux(
				fields.rho[params.L_max_global],
				fields.v_z[params.L_max_global],
				grid.dr[params.L_max_global],
				params.M_max
				));

	printf("Thrust : %f \n", GetThrust(
				fields.rho[params.L_max_global],
				fields.v_z[params.L_max_global],
				fields.p[params.L_max_global],
				fields.H_r[params.L_max_global],
				fields.H_phi[params.L_max_global],
				fields.H_z[params.L_max_global],
				grid.dr[params.L_max_global],
				params.M_max
				));
        if (params.animate) {
            printf("Total animation frames generated: %d\n", frame_count);
        }
    }

    // Write final output file (always written, regardless of animation mode)
    WriteFinalOutput(fields, grid, domain, params, global_fields_anim, global_grid_anim,
                     step_count, frame_count, t);

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
