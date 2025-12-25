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

    params.T = 50.0;
    params.dt = 0.0000125;

    params.L_max_global = 800;
    params.L_end = 265;
    params.M_max = 400;

    params.dz = 1.0 / params.L_max_global;
    params.dy = 1.0 / params.M_max;

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

    // ========================================================================
    // Setup 2D Domain Decomposition
    // ========================================================================
    Setup2DDecomposition(domain, params);
    
    if (domain.rank == 0) {
        printf("\n========================================\n");
        printf("2D Domain Decomposition Configuration\n");
        printf("========================================\n");
        printf("Process grid: %d x %d (L x M)\n", domain.dims[0], domain.dims[1]);
        printf("Global domain: %d x %d\n", params.L_max_global, params.M_max + 1);
        printf("========================================\n\n");
    }
    
    // Print detailed decomposition info (for debugging)
    if (domain.rank == 0) {
        PrintDecompositionInfo(domain, params);
    }
    MPI_Barrier(GetCartComm(domain));

    double begin, end, total;

    // ========================================================================
    // Allocate arrays with 2D local dimensions
    // ========================================================================
    PhysicalFields fields;
    ConservativeVars u, u0;
    GridGeometry grid;

    // Note: Now using local_L_with_ghosts x local_M_with_ghosts
    AllocateFields(fields, domain.local_L_with_ghosts, domain.local_M_with_ghosts);
    AllocateConservative(u, domain.local_L_with_ghosts, domain.local_M_with_ghosts);
    AllocateConservative(u0, domain.local_L_with_ghosts, domain.local_M_with_ghosts);

    MemoryAllocation2D(grid.r, domain.local_L_with_ghosts, domain.local_M_with_ghosts);
    MemoryAllocation2D(grid.r_z, domain.local_L_with_ghosts, domain.local_M_with_ghosts);
    grid.R = new double[domain.local_L_with_ghosts];
    grid.dr = new double[domain.local_L_with_ghosts];

    for (int l = 0; l < domain.local_L_with_ghosts; l++) {
        grid.R[l] = 0;
        grid.dr[l] = 0;
    }

    PreviousState prev_state;
    if (params.convergence_threshold > 0) {
        AllocatePreviousState(prev_state, domain.local_L_with_ghosts, domain.local_M_with_ghosts);
    }

    // ========================================================================
    // Initialize grid geometry with 2D local indexing
    // ========================================================================
    double r_0 = (R1(0) + R2(0)) / 2.0;

    for (int l = 0; l < domain.local_L_with_ghosts; l++) {
        int l_global = domain.l_start + l - 1;  // -1 for ghost cell offset
        double z = l_global * params.dz;

        grid.R[l] = R2(z) - R1(z);
        grid.dr[l] = grid.R[l] / params.M_max;

        for (int m = 0; m < domain.local_M_with_ghosts; m++) {
            int m_global = domain.m_start + m - 1;  // -1 for ghost cell offset
            double y = static_cast<double>(m_global) / params.M_max;  // Normalized [0,1]
            
            grid.r[l][m] = (1 - y) * R1(z) + y * R2(z);
            grid.r_z[l][m] = (1 - y) * DerR1(z) + y * DerR2(z);
        }
    }

    // ========================================================================
    // Initialize physical fields with 2D local indexing
    // ========================================================================
    #pragma omp parallel for collapse(2)
    for (int l = 1; l <= domain.local_L; l++) {
        for (int m = 1; m <= domain.local_M; m++) {
            int l_global = domain.l_start + l - 1;
            int m_global = domain.m_start + m - 1;

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
            for (int m = 0; m < domain.local_M_with_ghosts; m++) {
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
    InitializeConservativeVars2D(u0, fields, grid, domain);

    // ========================================================================
    // Initialize ghost cells for boundary processes
    // ========================================================================
    // Left boundary ghost cells (z=0)
    if (domain.is_left_boundary) {
        #pragma omp parallel for
        for (int m = 0; m < domain.local_M_with_ghosts; m++) {
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

    // Right boundary ghost cells (z=z_max)
    if (domain.is_right_boundary) {
        int l = domain.local_L;
        #pragma omp parallel for
        for (int m = 0; m < domain.local_M_with_ghosts; m++) {
            fields.rho[l + 1][m] = fields.rho[l][m];
            fields.v_z[l + 1][m] = fields.v_z[l][m];
            fields.v_r[l + 1][m] = fields.v_r[l][m];
            fields.v_phi[l + 1][m] = fields.v_phi[l][m];
            fields.H_phi[l + 1][m] = fields.H_phi[l][m];
            fields.H_z[l + 1][m] = fields.H_z[l][m];
            fields.H_r[l + 1][m] = fields.H_r[l][m];
            fields.e[l + 1][m] = fields.e[l][m];
            fields.p[l + 1][m] = fields.p[l][m];
            fields.P[l + 1][m] = fields.P[l][m];

            u0.u_1[l + 1][m] = fields.rho[l + 1][m] * grid.r[l + 1][m];
            u0.u_2[l + 1][m] = fields.rho[l + 1][m] * fields.v_z[l + 1][m] * grid.r[l + 1][m];
            u0.u_3[l + 1][m] = fields.rho[l + 1][m] * fields.v_r[l + 1][m] * grid.r[l + 1][m];
            u0.u_4[l + 1][m] = fields.rho[l + 1][m] * fields.v_phi[l + 1][m] * grid.r[l + 1][m];
            u0.u_5[l + 1][m] = fields.rho[l + 1][m] * fields.e[l + 1][m] * grid.r[l + 1][m];
            u0.u_6[l + 1][m] = fields.H_phi[l + 1][m];
            u0.u_7[l + 1][m] = fields.H_z[l + 1][m] * grid.r[l + 1][m];
            u0.u_8[l + 1][m] = fields.H_r[l + 1][m] * grid.r[l + 1][m];
        }
    }

    // Bottom boundary ghost cells (m=0)
    if (domain.is_down_boundary) {
        #pragma omp parallel for
        for (int l = 0; l < domain.local_L_with_ghosts; l++) {
            fields.rho[l][0] = fields.rho[l][1];
            fields.v_z[l][0] = fields.v_z[l][1];
            fields.v_r[l][0] = fields.v_r[l][1];
            fields.v_phi[l][0] = fields.v_phi[l][1];
            fields.H_phi[l][0] = fields.H_phi[l][1];
            fields.H_z[l][0] = fields.H_z[l][1];
            fields.H_r[l][0] = fields.H_r[l][1];
            fields.e[l][0] = fields.e[l][1];
            fields.p[l][0] = fields.p[l][1];
            fields.P[l][0] = fields.P[l][1];

            u0.u_1[l][0] = u0.u_1[l][1];
            u0.u_2[l][0] = u0.u_2[l][1];
            u0.u_3[l][0] = u0.u_3[l][1];
            u0.u_4[l][0] = u0.u_4[l][1];
            u0.u_5[l][0] = u0.u_5[l][1];
            u0.u_6[l][0] = u0.u_6[l][1];
            u0.u_7[l][0] = u0.u_7[l][1];
            u0.u_8[l][0] = u0.u_8[l][1];
        }
    }

    // Top boundary ghost cells (m=M_max)
    if (domain.is_up_boundary) {
        int m = domain.local_M;
        #pragma omp parallel for
        for (int l = 0; l < domain.local_L_with_ghosts; l++) {
            fields.rho[l][m + 1] = fields.rho[l][m];
            fields.v_z[l][m + 1] = fields.v_z[l][m];
            fields.v_r[l][m + 1] = fields.v_r[l][m];
            fields.v_phi[l][m + 1] = fields.v_phi[l][m];
            fields.H_phi[l][m + 1] = fields.H_phi[l][m];
            fields.H_z[l][m + 1] = fields.H_z[l][m];
            fields.H_r[l][m + 1] = fields.H_r[l][m];
            fields.e[l][m + 1] = fields.e[l][m];
            fields.p[l][m + 1] = fields.p[l][m];
            fields.P[l][m + 1] = fields.P[l][m];

            u0.u_1[l][m + 1] = u0.u_1[l][m];
            u0.u_2[l][m + 1] = u0.u_2[l][m];
            u0.u_3[l][m + 1] = u0.u_3[l][m];
            u0.u_4[l][m + 1] = u0.u_4[l][m];
            u0.u_5[l][m + 1] = u0.u_5[l][m];
            u0.u_6[l][m + 1] = u0.u_6[l][m];
            u0.u_7[l][m + 1] = u0.u_7[l][m];
            u0.u_8[l][m + 1] = u0.u_8[l][m];
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
        MemoryAllocation2D(global_grid_anim.r, params.L_max_global + 1, params.M_max + 1);

        printf("Animation enabled: will output every %d steps\n",
               params.animation_frequency);
        printf("Output format: %s\n", params.output_format.c_str());
    }

    // Output initial conditions as frame 0 if animation is enabled
    if (params.animate) {
        GatherResultsToRank0_2D(fields, grid, domain, params, global_fields_anim,
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

    // ========================================================================
    // Main time loop using 2D decomposition functions
    // ========================================================================
    MPI_Comm cart_comm = GetCartComm(domain);
    
    while (t < params.T && !converged) {
        // Exchange ghost cells in both L and M directions
        ExchangeGhostCellsConservative2D(u0, domain, params);
        ExchangeGhostCellsPhysical2D(fields, domain, params);

        // Compute one time step
        ComputeTimeStep2D(u, u0, fields, grid, domain, params);

        // Update physical fields in interior
        UpdatePhysicalFields2D(fields, u, grid, domain, params.gamma);

        // Apply boundary conditions based on 2D position
        ApplyBoundaryConditions2D(fields, u, grid, domain, params, r_0);

        // Update physical fields including boundaries
        UpdatePhysicalFields2D(fields, u, grid, domain, params.gamma);

        // Copy u to u0
        CopyConservativeVars2D(u0, u, domain);

        // Convergence check
        if (params.convergence_threshold > 0 &&
            step_count % params.check_frequency == 0) {
            double change = ComputeSolutionChange2D(fields, prev_state, domain);

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
            for (int l = 1; l <= domain.local_L; l++) {
                for (int m = 1; m <= domain.local_M; m++) {
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
            double max_wave_speed = ComputeMaxWaveSpeed2D(fields, domain, params.gamma);
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

        // Debug output (only from rank 0, and only if it owns the check point)
        if (step_count % 1000 == 0) {
            int check_l_global = 20;
            int check_m_global = 40;
            
            // Check if this process owns the debug point
            bool owns_point = (check_l_global >= domain.l_start && check_l_global <= domain.l_end &&
                              check_m_global >= domain.m_start && check_m_global <= domain.m_end);
            
            if (owns_point) {
                int local_check_l = check_l_global - domain.l_start + 1;
                int local_check_m = check_m_global - domain.m_start + 1;
                printf("Rank %d: t=%lf\trho=%lf\tv_z=%lf\tv_phi=%lf\te=%lf\tH_phi=%lf\n", 
                       domain.rank, t,
                       fields.rho[local_check_l][local_check_m],
                       fields.v_z[local_check_l][local_check_m],
                       fields.v_phi[local_check_l][local_check_m],
                       fields.e[local_check_l][local_check_m],
                       fields.H_phi[local_check_l][local_check_m]);
            }
        }

        // Animation output
        if (params.animate && step_count % params.animation_frequency == 0) {
            GatherResultsToRank0_2D(fields, grid, domain, params, global_fields_anim,
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
        if (params.animate) {
            printf("Total animation frames generated: %d\n", frame_count);
        }
    }

    // Write final output file
    // Note: WriteFinalOutput needs to be updated for 2D - for now use direct gather
    {
        PhysicalFields global_fields;
        GridGeometry global_grid;

        if (domain.rank == 0) {
            AllocateFields(global_fields, params.L_max_global + 1, params.M_max + 1);
            MemoryAllocation2D(global_grid.r, params.L_max_global + 1, params.M_max + 1);
        }

        GatherResultsToRank0_2D(fields, grid, domain, params, global_fields, global_grid);

        if (domain.rank == 0) {
            std::string filename = GenerateOutputFilename(
                params.output_format, frame_count, domain.size, omp_get_max_threads(),
                params.filename_template, params.L_max_global, params.M_max, params.dt);

            if (params.output_format == "vtk") {
                WriteVtk(filename.c_str(), params.L_max_global, params.M_max,
                         params.dz, global_grid.r, global_fields.rho,
                         global_fields.v_z, global_fields.v_r, global_fields.v_phi,
                         global_fields.e, global_fields.H_z, global_fields.H_r,
                         global_fields.H_phi, params.output_dir);
            } else if (params.output_format == "plt") {
                WritePlt(filename.c_str(), params.L_max_global, params.M_max,
                         params.dz, global_grid.r, global_fields.rho,
                         global_fields.v_z, global_fields.v_r, global_fields.v_phi,
                         global_fields.e, global_fields.H_z, global_fields.H_r,
                         global_fields.H_phi, params.output_dir);
            }

            printf("Final output written to: %s/%s\n", params.output_dir.c_str(),
                   filename.c_str());

            DeallocateFields(global_fields, params.L_max_global + 1);
            MemoryClearing2D(global_grid.r, params.L_max_global + 1);
        }
    }

    // Clean up animation arrays if allocated
    if (domain.rank == 0 && params.animate) {
        DeallocateFields(global_fields_anim, params.L_max_global + 1);
        MemoryClearing2D(global_grid_anim.r, params.L_max_global + 1);
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

    // Free the Cartesian communicator
    MPI_Comm_free(&cart_comm);
    
    MPI_Finalize();
    return 0;
}
