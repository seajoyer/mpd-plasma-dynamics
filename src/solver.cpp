#include "solver.hpp"
#include <cmath>
#include <omp.h>

void initialize_conservative_vars(ConservativeVars& u0, const PhysicalFields& fields,
                                  const GridGeometry& grid, int local_L_with_ghosts, int M_max) {
    #pragma omp parallel for collapse(2)
    for (int l = 1; l < local_L_with_ghosts - 1; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            u0.u_1[l][m] = fields.rho[l][m] * grid.r[l][m];
            u0.u_2[l][m] = fields.rho[l][m] * fields.v_z[l][m] * grid.r[l][m];
            u0.u_3[l][m] = fields.rho[l][m] * fields.v_r[l][m] * grid.r[l][m];
            u0.u_4[l][m] = fields.rho[l][m] * fields.v_phi[l][m] * grid.r[l][m];
            u0.u_5[l][m] = fields.rho[l][m] * fields.e[l][m] * grid.r[l][m];
            u0.u_6[l][m] = fields.H_phi[l][m];
            u0.u_7[l][m] = fields.H_z[l][m] * grid.r[l][m];
            u0.u_8[l][m] = fields.H_r[l][m] * grid.r[l][m];
        }
    }
}

void compute_time_step(ConservativeVars& u, const ConservativeVars& u0,
                      const PhysicalFields& fields, const GridGeometry& grid,
                      const DomainInfo& domain, const SimulationParams& params) {
    
    const int local_L = domain.local_L;
    const int M_max = params.M_max;
    const double dz = params.dz;
    const double dt = params.dt;
    
    // Lax-Friedrichs scheme for interior cells
    #pragma omp parallel for collapse(2)
    for (int l = 1; l < local_L + 1; l++) {
        for (int m = 1; m < M_max; m++) {
            u.u_1[l][m] = 0.25 * (u0.u_1[l + 1][m] + u0.u_1[l - 1][m] + u0.u_1[l][m + 1] + u0.u_1[l][m - 1]) +
                        dt * (0 - 
                              (u0.u_1[l + 1][m] * fields.v_z[l + 1][m] - u0.u_1[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u0.u_1[l][m + 1] * fields.v_r[l][m + 1] - u0.u_1[l][m - 1] * fields.v_r[l][m - 1]) / (2 * grid.dr[l]));

            u.u_2[l][m] = 0.25 * (u0.u_2[l + 1][m] + u0.u_2[l - 1][m] + u0.u_2[l][m + 1] + u0.u_2[l][m - 1]) +
                        dt * (((pow(fields.H_z[l + 1][m], 2) - fields.P[l + 1][m]) * grid.r[l + 1][m] - 
                               (pow(fields.H_z[l - 1][m], 2) - fields.P[l - 1][m]) * grid.r[l - 1][m]) / (2 * dz) + 
                              ((fields.H_z[l][m + 1] * fields.H_r[l][m + 1]) * grid.r[l][m + 1] - 
                               (fields.H_z[l][m - 1] * fields.H_r[l][m - 1]) * grid.r[l][m - 1]) / (2 * grid.dr[l]) - 
                              (u0.u_2[l + 1][m] * fields.v_z[l + 1][m] - u0.u_2[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u0.u_2[l][m + 1] * fields.v_r[l][m + 1] - u0.u_2[l][m - 1] * fields.v_r[l][m - 1]) / (2 * grid.dr[l]));

            u.u_3[l][m] = 0.25 * (u0.u_3[l + 1][m] + u0.u_3[l - 1][m] + u0.u_3[l][m + 1] + u0.u_3[l][m - 1]) +
                        dt * ((fields.rho[l][m] * pow(fields.v_phi[l][m], 2) + fields.P[l][m] - pow(fields.H_phi[l][m], 2)) + 
                              (fields.H_z[l + 1][m] * fields.H_r[l + 1][m] * grid.r[l + 1][m] - 
                               fields.H_z[l - 1][m] * fields.H_r[l - 1][m] * grid.r[l - 1][m]) / (2 * dz) + 
                              ((pow(fields.H_r[l][m + 1], 2) - fields.P[l][m + 1]) * grid.r[l][m + 1] - 
                               (pow(fields.H_r[l][m - 1], 2) - fields.P[l][m - 1]) * grid.r[l][m - 1]) / (2 * grid.dr[l]) - 
                              (u0.u_3[l + 1][m] * fields.v_z[l + 1][m] - u0.u_3[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u0.u_3[l][m + 1] * fields.v_r[l][m + 1] - u0.u_3[l][m - 1] * fields.v_r[l][m - 1]) / (2 * grid.dr[l]));

            u.u_4[l][m] = 0.25 * (u0.u_4[l + 1][m] + u0.u_4[l - 1][m] + u0.u_4[l][m + 1] + u0.u_4[l][m - 1]) +
                        dt * ((-fields.rho[l][m] * fields.v_r[l][m] * fields.v_phi[l][m] + fields.H_phi[l][m] * fields.H_r[l][m]) + 
                              (fields.H_phi[l + 1][m] * fields.H_z[l + 1][m] * grid.r[l + 1][m] - 
                               fields.H_phi[l - 1][m] * fields.H_z[l - 1][m] * grid.r[l - 1][m]) / (2 * dz) + 
                              (fields.H_phi[l][m + 1] * fields.H_r[l][m + 1] * grid.r[l][m + 1] - 
                               fields.H_phi[l][m - 1] * fields.H_r[l][m - 1] * grid.r[l][m - 1]) / (2 * grid.dr[l]) - 
                              (u0.u_4[l + 1][m] * fields.v_z[l + 1][m] - u0.u_4[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u0.u_4[l][m + 1] * fields.v_r[l][m + 1] - u0.u_4[l][m - 1] * fields.v_r[l][m - 1]) / (2 * grid.dr[l]));

            u.u_5[l][m] = 0.25 * (u0.u_5[l + 1][m] + u0.u_5[l - 1][m] + u0.u_5[l][m + 1] + u0.u_5[l][m - 1]) +
                        dt * (-fields.p[l][m] * ((fields.v_z[l + 1][m] * grid.r[l + 1][m] -
                                       fields.v_z[l - 1][m] * grid.r[l - 1][m]) / (2 * dz) + 
                                      (fields.v_r[l][m + 1] * grid.r[l][m + 1] - 
                                       fields.v_r[l][m - 1] * grid.r[l][m - 1]) / (2 * grid.dr[l])) - 
                              (u0.u_5[l + 1][m] * fields.v_z[l + 1][m] - u0.u_5[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u0.u_5[l][m + 1] * fields.v_r[l][m + 1] - u0.u_5[l][m - 1] * fields.v_r[l][m - 1]) / (2 * grid.dr[l]));

            u.u_6[l][m] = 0.25 * (u0.u_6[l + 1][m] + u0.u_6[l - 1][m] + u0.u_6[l][m + 1] + u0.u_6[l][m - 1]) +
                        dt * ((fields.H_z[l + 1][m] * fields.v_phi[l + 1][m] - 
                               fields.H_z[l - 1][m] * fields.v_phi[l - 1][m]) / (2 * dz) + 
                              (fields.H_r[l][m + 1] * fields.v_phi[l][m + 1] - 
                               fields.H_r[l][m - 1] * fields.v_phi[l][m - 1]) / (2 * grid.dr[l]) - 
                              (u0.u_6[l + 1][m] * fields.v_z[l + 1][m] - u0.u_6[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u0.u_6[l][m + 1] * fields.v_r[l][m + 1] - u0.u_6[l][m - 1] * fields.v_r[l][m - 1]) / (2 * grid.dr[l]));

            u.u_7[l][m] = 0.25 * (u0.u_7[l + 1][m] + u0.u_7[l - 1][m] + u0.u_7[l][m + 1] + u0.u_7[l][m - 1]) +
                        dt * ((fields.H_r[l][m + 1] * fields.v_z[l][m + 1] * grid.r[l][m + 1] - 
                               fields.H_r[l][m - 1] * fields.v_z[l][m - 1] * grid.r[l][m - 1]) / (2 * grid.dr[l]) - 
                              (u0.u_7[l][m + 1] * fields.v_r[l][m + 1] - u0.u_7[l][m - 1] * fields.v_r[l][m - 1]) / (2 * grid.dr[l]));

            u.u_8[l][m] = 0.25 * (u0.u_8[l + 1][m] + u0.u_8[l - 1][m] + u0.u_8[l][m + 1] + u0.u_8[l][m - 1]) +
                        dt * ((fields.H_z[l + 1][m] * fields.v_r[l + 1][m] * grid.r[l + 1][m] - 
                               fields.H_z[l - 1][m] * fields.v_r[l - 1][m] * grid.r[l - 1][m]) / (2 * dz) - 
                              (u0.u_8[l + 1][m] * fields.v_z[l + 1][m] - u0.u_8[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz));
        }
    }
}

void update_physical_fields(PhysicalFields& fields, const ConservativeVars& u,
                           const GridGeometry& grid, int local_L_with_ghosts, int M_max,
                           double gamma) {
    #pragma omp parallel for collapse(2)
    for (int l = 1; l < local_L_with_ghosts - 1; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            // Prevent division by zero
            double u1_safe = (std::abs(u.u_1[l][m]) < 1e-15) ? 1e-15 : u.u_1[l][m];
            double r_safe = (std::abs(grid.r[l][m]) < 1e-15) ? 1e-15 : grid.r[l][m];
            
            fields.rho[l][m] = u.u_1[l][m] / r_safe;
            fields.v_z[l][m] = u.u_2[l][m] / u1_safe;
            fields.v_r[l][m] = u.u_3[l][m] / u1_safe;
            fields.v_phi[l][m] = u.u_4[l][m] / u1_safe;

            fields.H_phi[l][m] = u.u_6[l][m];
            fields.H_z[l][m] = u.u_7[l][m] / r_safe;
            fields.H_r[l][m] = u.u_8[l][m] / r_safe;

            fields.e[l][m] = u.u_5[l][m] / u1_safe;
            fields.p[l][m] = (gamma - 1) * fields.rho[l][m] * fields.e[l][m];
            fields.P[l][m] = fields.p[l][m] + 0.5 * (pow(fields.H_z[l][m], 2) + 
                                                      pow(fields.H_r[l][m], 2) + 
                                                      pow(fields.H_phi[l][m], 2));
        }
    }
}

void copy_conservative_vars(ConservativeVars& u0, const ConservativeVars& u,
                           int local_L_with_ghosts, int M_max) {
    #pragma omp parallel for collapse(2)
    for (int l = 0; l < local_L_with_ghosts; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            u0.u_1[l][m] = u.u_1[l][m];
            u0.u_2[l][m] = u.u_2[l][m];
            u0.u_3[l][m] = u.u_3[l][m];
            u0.u_4[l][m] = u.u_4[l][m];
            u0.u_5[l][m] = u.u_5[l][m];
            u0.u_6[l][m] = u.u_6[l][m];
            u0.u_7[l][m] = u.u_7[l][m];
            u0.u_8[l][m] = u.u_8[l][m];
        }
    }
}
