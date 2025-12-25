#include "boundary.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

void ApplyBoundaryConditions2D(PhysicalFields& fields, ConservativeVars& u, 
                                const GridGeometry& grid, const DomainInfo& domain,
                                const SimulationParams& params, double r_0) {
    
    const int local_L = domain.local_L;
    const int local_M = domain.local_M;
    const int L_end = params.L_end;
    const int L_max_global = params.L_max_global;
    const double beta = params.beta;
    const double gamma = params.gamma;
    const double H_z0 = params.H_z0;
    const double dz = params.dz;
    
    // ========================================================================
    // LEFT BOUNDARY (z = 0, inlet) - only for processes at left edge
    // ========================================================================
    if (domain.is_left_boundary) {
        #pragma omp parallel for
        for (int m_local = 1; m_local <= local_M; m_local++) {
            int l = 1;  // First interior cell in local coordinates
            
            fields.rho[l][m_local] = 1.0;
            fields.v_phi[l][m_local] = 0;
            fields.v_z[l][m_local] = u.u_2[2][m_local] / (fields.rho[l][m_local] * grid.r[l][m_local]);
            fields.v_r[l][m_local] = 0;
            fields.H_phi[l][m_local] = r_0 / grid.r[l][m_local];
            fields.H_z[l][m_local] = H_z0;
            fields.H_r[l][m_local] = 0;
            fields.e[l][m_local] = beta / (2.0 * (gamma - 1.0)) * pow(fields.rho[l][m_local], gamma - 1.0);

            u.u_1[l][m_local] = fields.rho[l][m_local] * grid.r[l][m_local];
            u.u_2[l][m_local] = fields.rho[l][m_local] * fields.v_z[l][m_local] * grid.r[l][m_local];
            u.u_3[l][m_local] = fields.rho[l][m_local] * fields.v_r[l][m_local] * grid.r[l][m_local];
            u.u_4[l][m_local] = fields.rho[l][m_local] * fields.v_phi[l][m_local] * grid.r[l][m_local];
            u.u_5[l][m_local] = fields.rho[l][m_local] * fields.e[l][m_local] * grid.r[l][m_local];
            u.u_6[l][m_local] = fields.H_phi[l][m_local];
            u.u_7[l][m_local] = fields.H_z[l][m_local] * grid.r[l][m_local];
            u.u_8[l][m_local] = fields.H_r[l][m_local] * grid.r[l][m_local];
            
            // Update left ghost cell to match boundary
            fields.rho[0][m_local] = fields.rho[1][m_local];
            fields.v_z[0][m_local] = fields.v_z[1][m_local];
            fields.v_r[0][m_local] = fields.v_r[1][m_local];
            fields.v_phi[0][m_local] = fields.v_phi[1][m_local];
            fields.H_phi[0][m_local] = fields.H_phi[1][m_local];
            fields.H_z[0][m_local] = fields.H_z[1][m_local];
            fields.H_r[0][m_local] = fields.H_r[1][m_local];
            fields.e[0][m_local] = fields.e[1][m_local];
            
            u.u_1[0][m_local] = u.u_1[1][m_local];
            u.u_2[0][m_local] = u.u_2[1][m_local];
            u.u_3[0][m_local] = u.u_3[1][m_local];
            u.u_4[0][m_local] = u.u_4[1][m_local];
            u.u_5[0][m_local] = u.u_5[1][m_local];
            u.u_6[0][m_local] = u.u_6[1][m_local];
            u.u_7[0][m_local] = u.u_7[1][m_local];
            u.u_8[0][m_local] = u.u_8[1][m_local];
        }
    }

    // ========================================================================
    // UP BOUNDARY (outer wall, m = M_max) - only for processes at top edge
    // ========================================================================
    if (domain.is_up_boundary) {
        int m_local = local_M;  // Last interior row in local coordinates
        
        #pragma omp parallel for
        for (int l = 1; l <= local_L; l++) {
            fields.rho[l][m_local] = fields.rho[l][m_local - 1];
            fields.v_z[l][m_local] = fields.v_z[l][m_local - 1];
            fields.v_r[l][m_local] = fields.v_z[l][m_local] * grid.r_z[l][m_local];
            fields.v_phi[l][m_local] = fields.v_phi[l][m_local - 1];
            fields.e[l][m_local] = fields.e[l][m_local - 1];
            fields.H_phi[l][m_local] = fields.H_phi[l][m_local - 1];
            fields.H_z[l][m_local] = fields.H_z[l][m_local - 1];
            fields.H_r[l][m_local] = fields.H_z[l][m_local] * grid.r_z[l][m_local];

            u.u_1[l][m_local] = fields.rho[l][m_local] * grid.r[l][m_local];
            u.u_2[l][m_local] = fields.rho[l][m_local] * fields.v_z[l][m_local] * grid.r[l][m_local];
            u.u_3[l][m_local] = fields.rho[l][m_local] * fields.v_r[l][m_local] * grid.r[l][m_local];
            u.u_4[l][m_local] = fields.rho[l][m_local] * fields.v_phi[l][m_local] * grid.r[l][m_local];
            u.u_5[l][m_local] = fields.rho[l][m_local] * fields.e[l][m_local] * grid.r[l][m_local];
            u.u_6[l][m_local] = fields.H_phi[l][m_local];
            u.u_7[l][m_local] = fields.H_z[l][m_local] * grid.r[l][m_local];
            u.u_8[l][m_local] = fields.H_r[l][m_local] * grid.r[l][m_local];
            
            // Update top ghost cell
            fields.rho[l][m_local + 1] = fields.rho[l][m_local];
            fields.v_z[l][m_local + 1] = fields.v_z[l][m_local];
            fields.v_r[l][m_local + 1] = fields.v_r[l][m_local];
            fields.v_phi[l][m_local + 1] = fields.v_phi[l][m_local];
            fields.e[l][m_local + 1] = fields.e[l][m_local];
            fields.H_phi[l][m_local + 1] = fields.H_phi[l][m_local];
            fields.H_z[l][m_local + 1] = fields.H_z[l][m_local];
            fields.H_r[l][m_local + 1] = fields.H_r[l][m_local];
            
            u.u_1[l][m_local + 1] = u.u_1[l][m_local];
            u.u_2[l][m_local + 1] = u.u_2[l][m_local];
            u.u_3[l][m_local + 1] = u.u_3[l][m_local];
            u.u_4[l][m_local + 1] = u.u_4[l][m_local];
            u.u_5[l][m_local + 1] = u.u_5[l][m_local];
            u.u_6[l][m_local + 1] = u.u_6[l][m_local];
            u.u_7[l][m_local + 1] = u.u_7[l][m_local];
            u.u_8[l][m_local + 1] = u.u_8[l][m_local];
        }
    }

    // ========================================================================
    // DOWN BOUNDARY (inner wall/axis, m = 0) - only for processes at bottom edge
    // ========================================================================
    if (domain.is_down_boundary) {
        int m_local = 1;  // First interior row in local coordinates
        int m_global = domain.m_start;
        
        // Only apply if this process has global m = 0
        if (m_global == 0) {
            // Down boundary condition for l <= L_end (reflective wall)
            #pragma omp parallel for
            for (int l = 1; l <= local_L; l++) {
                int l_global = domain.l_start + l - 1;
                
                if (l_global >= 1 && l_global <= L_end) {
                    fields.rho[l][m_local] = fields.rho[l][m_local + 1];
                    fields.v_z[l][m_local] = fields.v_z[l][m_local + 1];
                    fields.v_r[l][m_local] = fields.v_z[l][m_local + 1] * grid.r_z[l][m_local + 1];
                    fields.v_phi[l][m_local] = fields.v_phi[l][m_local + 1];
                    fields.e[l][m_local] = fields.e[l][m_local + 1];
                    fields.H_phi[l][m_local] = fields.H_phi[l][m_local + 1];
                    fields.H_z[l][m_local] = fields.H_z[l][m_local + 1];
                    fields.H_r[l][m_local] = fields.H_z[l][m_local + 1] * grid.r_z[l][m_local + 1];

                    u.u_1[l][m_local] = fields.rho[l][m_local] * grid.r[l][m_local];
                    u.u_2[l][m_local] = fields.rho[l][m_local] * fields.v_z[l][m_local] * grid.r[l][m_local];
                    u.u_3[l][m_local] = fields.rho[l][m_local] * fields.v_r[l][m_local] * grid.r[l][m_local];
                    u.u_4[l][m_local] = fields.rho[l][m_local] * fields.v_phi[l][m_local] * grid.r[l][m_local];
                    u.u_5[l][m_local] = fields.rho[l][m_local] * fields.e[l][m_local] * grid.r[l][m_local];
                    u.u_6[l][m_local] = fields.H_phi[l][m_local];
                    u.u_7[l][m_local] = fields.H_z[l][m_local] * grid.r[l][m_local];
                    u.u_8[l][m_local] = fields.H_r[l][m_local] * grid.r[l][m_local];
                    
                    // Update bottom ghost cell
                    fields.rho[l][0] = fields.rho[l][m_local];
                    fields.v_z[l][0] = fields.v_z[l][m_local];
                    fields.v_r[l][0] = fields.v_r[l][m_local];
                    fields.v_phi[l][0] = fields.v_phi[l][m_local];
                    fields.e[l][0] = fields.e[l][m_local];
                    fields.H_phi[l][0] = fields.H_phi[l][m_local];
                    fields.H_z[l][0] = fields.H_z[l][m_local];
                    fields.H_r[l][0] = fields.H_r[l][m_local];
                    
                    u.u_1[l][0] = u.u_1[l][m_local];
                    u.u_2[l][0] = u.u_2[l][m_local];
                    u.u_3[l][0] = u.u_3[l][m_local];
                    u.u_4[l][0] = u.u_4[l][m_local];
                    u.u_5[l][0] = u.u_5[l][m_local];
                    u.u_6[l][0] = u.u_6[l][m_local];
                    u.u_7[l][0] = u.u_7[l][m_local];
                    u.u_8[l][0] = u.u_8[l][m_local];
                }
            }

            // Down boundary condition for l > L_end (characteristic-based outflow)
            #pragma omp parallel for
            for (int l = 1; l <= local_L; l++) {
                int l_global = domain.l_start + l - 1;
                
                if (l_global > L_end && l_global < L_max_global) {
                    int m = m_local;  // m = 0 in global, m = 1 in local
                    
                    u.u_1[l][m] = (0.25 * (u.u_1[l + 1][m] / grid.r[l + 1][m] + u.u_1[l - 1][m] / grid.r[l - 1][m] + 
                                            u.u_1[l][m + 1] / grid.r[l][m + 1] + u.u_1[l][m] / grid.r[l][m]) +
                                  params.dt * (0 - 
                                      (u.u_1[l + 1][m] / grid.r[l + 1][m] * fields.v_z[l + 1][m] - 
                                       u.u_1[l - 1][m] / grid.r[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                                      (u.u_1[l][m + 1] / grid.r[l][m + 1] * fields.v_r[l][m + 1] - 
                                       u.u_1[l][m] / grid.r[l][m] * (fields.v_r[l][m + 1])) / (grid.dr[l]))
                                  ) * grid.r[l][m];

                    u.u_2[l][m] = (0.25 * (u.u_2[l + 1][m] / grid.r[l + 1][m] + u.u_2[l - 1][m] / grid.r[l - 1][m] + 
                                            u.u_2[l][m + 1] / grid.r[l][m + 1] + u.u_2[l][m] / grid.r[l][m]) +
                                  params.dt * (((pow(fields.H_z[l + 1][m], 2) - fields.P[l + 1][m]) - 
                                       (pow(fields.H_z[l - 1][m], 2) - fields.P[l - 1][m])) / (2 * dz) + 
                                      ((fields.H_z[l][m + 1] * fields.H_r[l][m + 1]) - 
                                       (fields.H_z[l][m] * (fields.H_r[l][m]))) / (grid.dr[l]) - 
                                      (u.u_2[l + 1][m] / grid.r[l + 1][m] * fields.v_z[l + 1][m] - 
                                       u.u_2[l - 1][m] / grid.r[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                                      (u.u_2[l][m + 1] / grid.r[l][m + 1] * fields.v_r[l][m + 1] - 
                                       u.u_2[l][m] / grid.r[l][m] * (fields.v_r[l][m])) / (grid.dr[l]))
                                  ) * grid.r[l][m];

                    u.u_3[l][m] = 0;
                    u.u_4[l][m] = 0;

                    u.u_5[l][m] = (0.25 * (u.u_5[l + 1][m] / grid.r[l + 1][m] + u.u_5[l - 1][m] / grid.r[l - 1][m] + 
                                            u.u_5[l][m + 1] / grid.r[l][m + 1] + u.u_5[l][m] / grid.r[l][m]) +
                                  params.dt * (-fields.p[l][m] * ((fields.v_z[l + 1][m] -
                                               fields.v_z[l - 1][m]) / (2 * dz) + 
                                              (fields.v_r[l][m + 1] - 
                                               (fields.v_r[l][m])) / (grid.dr[l])) - 
                                      (u.u_5[l + 1][m] / grid.r[l + 1][m] * fields.v_z[l + 1][m] - 
                                       u.u_5[l - 1][m] / grid.r[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                                      (u.u_5[l][m + 1] / grid.r[l][m + 1] * fields.v_r[l][m + 1] - 
                                       u.u_5[l][m] / grid.r[l][m] * (fields.v_r[l][m])) / (grid.dr[l]))
                                  ) * grid.r[l][m];

                    u.u_6[l][m] = 0;

                    u.u_7[l][m] = (0.25 * (u.u_7[l + 1][m] / grid.r[l + 1][m] + u.u_7[l - 1][m] / grid.r[l - 1][m] + 
                                            u.u_7[l][m + 1] / grid.r[l][m + 1] + u.u_7[l][m] / grid.r[l][m]) +
                                  params.dt * ((fields.H_r[l][m + 1] * fields.v_z[l][m + 1] - 
                                       (fields.H_r[l][m]) * fields.v_z[l][m]) / (grid.dr[l]) - 
                                      (u.u_7[l][m + 1] / grid.r[l][m + 1] * fields.v_r[l][m + 1] - 
                                       u.u_7[l][m] / grid.r[l][m] * (fields.v_r[l][m])) / (grid.dr[l]))
                                  ) * grid.r[l][m];

                    u.u_8[l][m] = 0;
                    
                    // Update bottom ghost cell
                    u.u_1[l][0] = u.u_1[l][m];
                    u.u_2[l][0] = u.u_2[l][m];
                    u.u_3[l][0] = u.u_3[l][m];
                    u.u_4[l][0] = u.u_4[l][m];
                    u.u_5[l][0] = u.u_5[l][m];
                    u.u_6[l][0] = u.u_6[l][m];
                    u.u_7[l][0] = u.u_7[l][m];
                    u.u_8[l][0] = u.u_8[l][m];
                }
            }
        }
    }

    // ========================================================================
    // RIGHT BOUNDARY (z = z_max, outlet) - only for processes at right edge
    // ========================================================================
    if (domain.is_right_boundary) {
        int l = local_L;  // Last interior column in local coordinates
        
        #pragma omp parallel for
        for (int m_local = 1; m_local <= local_M; m_local++) {
            u.u_1[l][m_local] = u.u_1[l - 1][m_local];
            u.u_2[l][m_local] = u.u_2[l - 1][m_local];
            u.u_3[l][m_local] = u.u_3[l - 1][m_local];
            u.u_4[l][m_local] = u.u_4[l - 1][m_local];
            u.u_5[l][m_local] = u.u_5[l - 1][m_local];
            u.u_6[l][m_local] = u.u_6[l - 1][m_local];
            u.u_7[l][m_local] = u.u_7[l - 1][m_local];
            u.u_8[l][m_local] = u.u_8[l - 1][m_local];
            
            // Update right ghost cell
            u.u_1[l + 1][m_local] = u.u_1[l][m_local];
            u.u_2[l + 1][m_local] = u.u_2[l][m_local];
            u.u_3[l + 1][m_local] = u.u_3[l][m_local];
            u.u_4[l + 1][m_local] = u.u_4[l][m_local];
            u.u_5[l + 1][m_local] = u.u_5[l][m_local];
            u.u_6[l + 1][m_local] = u.u_6[l][m_local];
            u.u_7[l + 1][m_local] = u.u_7[l][m_local];
            u.u_8[l + 1][m_local] = u.u_8[l][m_local];
            
            fields.rho[l + 1][m_local] = fields.rho[l][m_local];
            fields.v_z[l + 1][m_local] = fields.v_z[l][m_local];
            fields.v_r[l + 1][m_local] = fields.v_r[l][m_local];
            fields.v_phi[l + 1][m_local] = fields.v_phi[l][m_local];
            fields.H_phi[l + 1][m_local] = fields.H_phi[l][m_local];
            fields.H_z[l + 1][m_local] = fields.H_z[l][m_local];
            fields.H_r[l + 1][m_local] = fields.H_r[l][m_local];
            fields.e[l + 1][m_local] = fields.e[l][m_local];
        }
    }
}

// ============================================================================
// Legacy 1D version (kept for compatibility)
// ============================================================================

void ApplyBoundaryConditions(PhysicalFields& fields, ConservativeVars& u, 
                               const GridGeometry& grid, const DomainInfo& domain,
                               const SimulationParams& params, double r_0) {
    
    const int local_L = domain.local_L;
    const int M_max = params.M_max;
    const int L_end = params.L_end;
    const int L_max_global = params.L_max_global;
    const double beta = params.beta;
    const double gamma = params.gamma;
    const double H_z0 = params.H_z0;
    
    // Left boundary condition (only for rank 0)
    if (domain.rank == 0) {
        #pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            fields.rho[1][m] = 1.0;
            fields.v_phi[1][m] = 0;
            fields.v_z[1][m] = u.u_2[2][m] / (fields.rho[1][m] * grid.r[1][m]);
            fields.v_r[1][m] = 0;
            fields.H_phi[1][m] = r_0 / grid.r[1][m];
            fields.H_z[1][m] = H_z0;
            fields.H_r[1][m] = 0;
            fields.e[1][m] = beta / (2.0 * (gamma - 1.0)) * pow(fields.rho[1][m], gamma - 1.0);
        }

        #pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            u.u_1[1][m] = fields.rho[1][m] * grid.r[1][m];
            u.u_2[1][m] = fields.rho[1][m] * fields.v_z[1][m] * grid.r[1][m];
            u.u_3[1][m] = fields.rho[1][m] * fields.v_r[1][m] * grid.r[1][m];
            u.u_4[1][m] = fields.rho[1][m] * fields.v_phi[1][m] * grid.r[1][m];
            u.u_5[1][m] = fields.rho[1][m] * fields.e[1][m] * grid.r[1][m];
            u.u_6[1][m] = fields.H_phi[1][m];
            u.u_7[1][m] = fields.H_z[1][m] * grid.r[1][m];
            u.u_8[1][m] = fields.H_r[1][m] * grid.r[1][m];
        }
        
        // Also update left ghost cell to match boundary
        #pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            fields.rho[0][m] = fields.rho[1][m];
            fields.v_z[0][m] = fields.v_z[1][m];
            fields.v_r[0][m] = fields.v_r[1][m];
            fields.v_phi[0][m] = fields.v_phi[1][m];
            fields.H_phi[0][m] = fields.H_phi[1][m];
            fields.H_z[0][m] = fields.H_z[1][m];
            fields.H_r[0][m] = fields.H_r[1][m];
            fields.e[0][m] = fields.e[1][m];
            
            u.u_1[0][m] = u.u_1[1][m];
            u.u_2[0][m] = u.u_2[1][m];
            u.u_3[0][m] = u.u_3[1][m];
            u.u_4[0][m] = u.u_4[1][m];
            u.u_5[0][m] = u.u_5[1][m];
            u.u_6[0][m] = u.u_6[1][m];
            u.u_7[0][m] = u.u_7[1][m];
            u.u_8[0][m] = u.u_8[1][m];
        }
    }

    // Up boundary condition
    #pragma omp parallel for
    for (int l = 1; l < local_L + 1; l++) {
        fields.rho[l][M_max] = fields.rho[l][M_max - 1];
        fields.v_z[l][M_max] = fields.v_z[l][M_max - 1];
        fields.v_r[l][M_max] = fields.v_z[l][M_max] * grid.r_z[l][M_max];
        fields.v_phi[l][M_max] = fields.v_phi[l][M_max - 1];
        fields.e[l][M_max] = fields.e[l][M_max - 1];
        fields.H_phi[l][M_max] = fields.H_phi[l][M_max - 1];
        fields.H_z[l][M_max] = fields.H_z[l][M_max - 1];
        fields.H_r[l][M_max] = fields.H_z[l][M_max] * grid.r_z[l][M_max];

        u.u_1[l][M_max] = fields.rho[l][M_max] * grid.r[l][M_max];
        u.u_2[l][M_max] = fields.rho[l][M_max] * fields.v_z[l][M_max] * grid.r[l][M_max];
        u.u_3[l][M_max] = fields.rho[l][M_max] * fields.v_r[l][M_max] * grid.r[l][M_max];
        u.u_4[l][M_max] = fields.rho[l][M_max] * fields.v_phi[l][M_max] * grid.r[l][M_max];
        u.u_5[l][M_max] = fields.rho[l][M_max] * fields.e[l][M_max] * grid.r[l][M_max];
        u.u_6[l][M_max] = fields.H_phi[l][M_max];
        u.u_7[l][M_max] = fields.H_z[l][M_max] * grid.r[l][M_max];
        u.u_8[l][M_max] = fields.H_r[l][M_max] * grid.r[l][M_max];
    }

    // Down boundary condition l <= L_end
    // Check if this process contains cells in the [1, L_end] range
    int local_L_end_rel = -1;
    if (domain.l_start <= L_end && domain.l_end >= 1) {
        int L_end_in_domain = std::min(L_end, domain.l_end);
        local_L_end_rel = L_end_in_domain - domain.l_start + 1;
        
        for (int l = 1; l <= local_L_end_rel; l++) {
            int l_global = domain.l_start + l - 1;
            if (l_global >= 1 && l_global <= L_end) {
                fields.rho[l][0] = fields.rho[l][1];
                fields.v_z[l][0] = fields.v_z[l][1];
                fields.v_r[l][0] = fields.v_z[l][1] * grid.r_z[l][1];
                fields.v_phi[l][0] = fields.v_phi[l][1];
                fields.e[l][0] = fields.e[l][1];
                fields.H_phi[l][0] = fields.H_phi[l][1];
                fields.H_z[l][0] = fields.H_z[l][1];
                fields.H_r[l][0] = fields.H_z[l][1] * grid.r_z[l][1];

                u.u_1[l][0] = fields.rho[l][0] * grid.r[l][0];
                u.u_2[l][0] = fields.rho[l][0] * fields.v_z[l][0] * grid.r[l][0];
                u.u_3[l][0] = fields.rho[l][0] * fields.v_r[l][0] * grid.r[l][0];
                u.u_4[l][0] = fields.rho[l][0] * fields.v_phi[l][0] * grid.r[l][0];
                u.u_5[l][0] = fields.rho[l][0] * fields.e[l][0] * grid.r[l][0];
                u.u_6[l][0] = fields.H_phi[l][0];
                u.u_7[l][0] = fields.H_z[l][0] * grid.r[l][0];
                u.u_8[l][0] = fields.H_r[l][0] * grid.r[l][0];
            }
        }
    }

    // Down boundary condition l > L_end
    const double dz = params.dz;
    
    #pragma omp parallel for
    for (int l = 1; l < local_L + 1; l++) {
        int l_global = domain.l_start + l - 1;
        
        if (l_global > L_end && l_global < L_max_global) {
            int m = 0;
            
            u.u_1[l][m] = (0.25 * (u.u_1[l + 1][m] / grid.r[l + 1][m] + u.u_1[l - 1][m] / grid.r[l - 1][m] + 
                                    u.u_1[l][m + 1] / grid.r[l][m + 1] + u.u_1[l][m] / grid.r[l][m]) +
                          params.dt * (0 - 
                              (u.u_1[l + 1][m] / grid.r[l + 1][m] * fields.v_z[l + 1][m] - 
                               u.u_1[l - 1][m] / grid.r[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u.u_1[l][m + 1] / grid.r[l][m + 1] * fields.v_r[l][m + 1] - 
                               u.u_1[l][m] / grid.r[l][m] * (fields.v_r[l][1])) / (grid.dr[l]))
                          ) * grid.r[l][m];

            u.u_2[l][m] = (0.25 * (u.u_2[l + 1][m] / grid.r[l + 1][m] + u.u_2[l - 1][m] / grid.r[l - 1][m] + 
                                    u.u_2[l][m + 1] / grid.r[l][m + 1] + u.u_2[l][m] / grid.r[l][m]) +
                          params.dt * (((pow(fields.H_z[l + 1][m], 2) - fields.P[l + 1][m]) - 
                               (pow(fields.H_z[l - 1][m], 2) - fields.P[l - 1][m])) / (2 * dz) + 
                              ((fields.H_z[l][m + 1] * fields.H_r[l][m + 1]) - 
                               (fields.H_z[l][m] * (fields.H_r[l][m]))) / (grid.dr[l]) - 
                              (u.u_2[l + 1][m] / grid.r[l + 1][m] * fields.v_z[l + 1][m] - 
                               u.u_2[l - 1][m] / grid.r[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u.u_2[l][m + 1] / grid.r[l][m + 1] * fields.v_r[l][m + 1] - 
                               u.u_2[l][m] / grid.r[l][m] * (fields.v_r[l][m])) / (grid.dr[l]))
                          ) * grid.r[l][m];

            u.u_3[l][m] = 0;
            u.u_4[l][m] = 0;

            u.u_5[l][m] = (0.25 * (u.u_5[l + 1][m] / grid.r[l + 1][m] + u.u_5[l - 1][m] / grid.r[l - 1][m] + 
                                    u.u_5[l][m + 1] / grid.r[l][m + 1] + u.u_5[l][m] / grid.r[l][m]) +
                          params.dt * (-fields.p[l][m] * ((fields.v_z[l + 1][m] -
                                       fields.v_z[l - 1][m]) / (2 * dz) + 
                                      (fields.v_r[l][m + 1] - 
                                       (fields.v_r[l][m])) / (grid.dr[l])) - 
                              (u.u_5[l + 1][m] / grid.r[l + 1][m] * fields.v_z[l + 1][m] - 
                               u.u_5[l - 1][m] / grid.r[l - 1][m] * fields.v_z[l - 1][m]) / (2 * dz) - 
                              (u.u_5[l][m + 1] / grid.r[l][m + 1] * fields.v_r[l][m + 1] - 
                               u.u_5[l][m] / grid.r[l][m] * (fields.v_r[l][m])) / (grid.dr[l]))
                          ) * grid.r[l][m];

            u.u_6[l][m] = 0;

            u.u_7[l][m] = (0.25 * (u.u_7[l + 1][m] / grid.r[l + 1][m] + u.u_7[l - 1][m] / grid.r[l - 1][m] + 
                                    u.u_7[l][m + 1] / grid.r[l][m + 1] + u.u_7[l][m] / grid.r[l][m]) +
                          params.dt * ((fields.H_r[l][m + 1] * fields.v_z[l][m + 1] - 
                               (fields.H_r[l][m]) * fields.v_z[l][m]) / (grid.dr[l]) - 
                              (u.u_7[l][m + 1] / grid.r[l][m + 1] * fields.v_r[l][m + 1] - 
                               u.u_7[l][m] / grid.r[l][m] * (fields.v_r[l][m])) / (grid.dr[l]))
                          ) * grid.r[l][m];

            u.u_8[l][m] = 0;
        }
    }

    // Right boundary condition (only for last rank)
    if (domain.rank == domain.size - 1) {
        #pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            u.u_1[local_L][m] = u.u_1[local_L - 1][m];
            u.u_2[local_L][m] = u.u_2[local_L - 1][m];
            u.u_3[local_L][m] = u.u_3[local_L - 1][m];
            u.u_4[local_L][m] = u.u_4[local_L - 1][m];
            u.u_5[local_L][m] = u.u_5[local_L - 1][m];
            u.u_6[local_L][m] = u.u_6[local_L - 1][m];
            u.u_7[local_L][m] = u.u_7[local_L - 1][m];
            u.u_8[local_L][m] = u.u_8[local_L - 1][m];
        }
        
        // Also update right ghost cell to match boundary
        #pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            u.u_1[local_L + 1][m] = u.u_1[local_L][m];
            u.u_2[local_L + 1][m] = u.u_2[local_L][m];
            u.u_3[local_L + 1][m] = u.u_3[local_L][m];
            u.u_4[local_L + 1][m] = u.u_4[local_L][m];
            u.u_5[local_L + 1][m] = u.u_5[local_L][m];
            u.u_6[local_L + 1][m] = u.u_6[local_L][m];
            u.u_7[local_L + 1][m] = u.u_7[local_L][m];
            u.u_8[local_L + 1][m] = u.u_8[local_L][m];
            
            fields.rho[local_L + 1][m] = fields.rho[local_L][m];
            fields.v_z[local_L + 1][m] = fields.v_z[local_L][m];
            fields.v_r[local_L + 1][m] = fields.v_r[local_L][m];
            fields.v_phi[local_L + 1][m] = fields.v_phi[local_L][m];
            fields.H_phi[local_L + 1][m] = fields.H_phi[local_L][m];
            fields.H_z[local_L + 1][m] = fields.H_z[local_L][m];
            fields.H_r[local_L + 1][m] = fields.H_r[local_L][m];
            fields.e[local_L + 1][m] = fields.e[local_L][m];
        }
    }
}
