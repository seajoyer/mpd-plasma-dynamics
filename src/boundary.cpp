#include "boundary.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

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
    const double dz = params.dz;
    
    // Step geometry parameters (must match geometry.cpp)
    const double z_center = 0.31;
    const double transition_width = 0.015;
    const double z_start = z_center - transition_width;  // Start of step transition
    const double z_end = z_center + transition_width;    // End of step transition
    const double r_before = 0.2;   // Inner radius before step
    const double r_after = 0.005;  // Inner radius after step (narrow throat)
    
    // Global l indices for step transition region (RED ZONE - inlet)
    const int l_step_start = static_cast<int>(z_start / dz);
    const int l_step_end = static_cast<int>(z_end / dz);
    
    // Calculate number of m-cells that span the lateral gap for inlet
    const double dy = params.dy;
    const double R2_val = 0.8;  // Outer radius (constant)
    // m_gap where r(l, m_gap) = r_before at the step region
    const int m_gap = static_cast<int>((r_before - r_after) / (dy * (R2_val - r_after)));
    
    // Inflow parameters for RED zone (inlet at step)
    const double v_inflow = 0.1;  // Inlet velocity
    
    // =========================================================================
    // LEFT BOUNDARY CONDITION (z=0) - WALL / NO-FLOW
    // Only for rank 0 - same as original boundary.cpp
    // =========================================================================
    if (domain.rank == 0) {
        #pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            // Wall boundary - no flow through the left wall
            // Extrapolate density and energy from interior (Neumann BC)
            fields.rho[1][m] = fields.rho[2][m];
            fields.e[1][m] = fields.e[2][m];
            
            // Zero velocity - no flow through wall
            fields.v_z[1][m] = 0.0;
            fields.v_r[1][m] = 0.0;
            fields.v_phi[1][m] = 0.0;
            
            // Magnetic field: perfectly conducting wall
            // Normal component (H_z) is zero at the wall
            // Tangential components (H_r, H_phi) are extrapolated
            fields.H_z[1][m] = 0.0;  // No normal B-field at conducting wall
            fields.H_r[1][m] = fields.H_r[2][m];
            fields.H_phi[1][m] = fields.H_phi[2][m];
        }

        // Update conservative variables at left boundary
        #pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            u.u_1[1][m] = fields.rho[1][m] * grid.r[1][m];
            u.u_2[1][m] = 0.0;  // rho * v_z * r = 0 (no axial flow)
            u.u_3[1][m] = 0.0;  // rho * v_r * r = 0 (no radial flow)
            u.u_4[1][m] = 0.0;  // rho * v_phi * r = 0 (no azimuthal flow)
            u.u_5[1][m] = fields.rho[1][m] * fields.e[1][m] * grid.r[1][m];
            u.u_6[1][m] = fields.H_phi[1][m];
            u.u_7[1][m] = 0.0;  // H_z * r = 0
            u.u_8[1][m] = fields.H_r[1][m] * grid.r[1][m];
        }
        
        // Update left ghost cell (l=0) - reflecting wall conditions
        #pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            fields.rho[0][m] = fields.rho[1][m];
            fields.v_z[0][m] = -fields.v_z[2][m];  // Reflect v_z (normal component)
            fields.v_r[0][m] = fields.v_r[2][m];   // Extrapolate v_r (tangential)
            fields.v_phi[0][m] = fields.v_phi[2][m]; // Extrapolate v_phi (tangential)
            fields.H_phi[0][m] = fields.H_phi[1][m];
            fields.H_z[0][m] = -fields.H_z[2][m];  // Reflect H_z (normal component)
            fields.H_r[0][m] = fields.H_r[1][m];
            fields.e[0][m] = fields.e[1][m];
            
            u.u_1[0][m] = u.u_1[1][m];
            u.u_2[0][m] = -u.u_2[2][m];  // Reflect
            u.u_3[0][m] = u.u_3[2][m];
            u.u_4[0][m] = u.u_4[2][m];
            u.u_5[0][m] = u.u_5[1][m];
            u.u_6[0][m] = u.u_6[1][m];
            u.u_7[0][m] = -u.u_7[2][m];  // Reflect
            u.u_8[0][m] = u.u_8[1][m];
        }
    }

    // =========================================================================
    // UP BOUNDARY CONDITION (outer radius, m = M_max)
    // Same as boundary-pure.cpp
    // =========================================================================
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

    // =========================================================================
    // GREEN ZONE: DOWN BOUNDARY CONDITION - BEFORE STEP (l < l_step_start, m = 0)
    // Non-leakage (slip wall) condition - same as boundary-pure.cpp
    // This is the inner wall before the step begins
    // =========================================================================
    for (int l = 1; l < local_L + 1; l++) {
        int l_global = domain.l_start + l - 1;
        
        // Only apply to cells before the step transition starts
        if (l_global >= 1 && l_global < l_step_start) {
            // Non-leakage (slip wall) condition from boundary-pure.cpp
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

   // =========================================================================
    // RED ZONE: INLET AT THE STEP SURFACE ONLY
    // Applies inlet stream ONLY at the boundary interface (m = 0) 
    // for the axial range of the step transition.
    // =========================================================================
    for (int l = 1; l < local_L + 1; l++) {
        int l_global = domain.l_start + l - 1;
        
        // Only apply condition at the step transition region (RED zone)
        if (l_global >= l_step_start && l_global <= l_step_end) {
            
            // We apply the condition ONLY to the surface (m = 0)
            const int m = 0;

            // 1. Set constant density as requested
            fields.rho[l][m] = 1.0;
            
            // 2. Velocity: Plasma enters through the surface
            // v_r < 0 means flow is directed radially inward from the step face
            fields.v_phi[l][m] = 0.0;
            fields.v_z[l][m]   = v_inflow;  // Axial component downstream
            fields.v_r[l][m]   = 0;       // Radial inward velocity at the surface
            
            // 3. Magnetic field at inlet surface
            fields.H_phi[l][m] = r_0 / grid.r[l][m];
            fields.H_z[l][m]   = H_z0;
            fields.H_r[l][m]   = fields.H_z[l][m] * grid.r_z[l][m];
            
            // 4. Internal energy (based on the specified rho and beta)
            fields.e[l][m] = beta / (2.0 * (gamma - 1.0)) * pow(fields.rho[l][m], gamma - 1.0);

            // 5. Update conservative variables for the surface cell
            u.u_1[l][m] = fields.rho[l][m] * grid.r[l][m];
            u.u_2[l][m] = fields.rho[l][m] * fields.v_z[l][m] * grid.r[l][m];
            u.u_3[l][m] = fields.rho[l][m] * fields.v_r[l][m] * grid.r[l][m];
            u.u_4[l][m] = fields.rho[l][m] * fields.v_phi[l][m] * grid.r[l][m];
            u.u_5[l][m] = fields.rho[l][m] * fields.e[l][m] * grid.r[l][m];
            u.u_6[l][m] = fields.H_phi[l][m];
            u.u_7[l][m] = fields.H_z[l][m] * grid.r[l][m];
            u.u_8[l][m] = fields.H_r[l][m] * grid.r[l][m];
        }
    } 
    // =========================================================================
    // REGION BETWEEN STEP AND L_END (l_step_end < l <= L_end, m = 0)
    // Non-leakage (slip wall) condition - inner wall after the step narrows
    // =========================================================================
    for (int l = 1; l < local_L + 1; l++) {
        int l_global = domain.l_start + l - 1;
        
        // Apply to cells after the step transition but before L_end
        if (l_global > l_step_end && l_global <= L_end) {
            // Non-leakage (slip wall) condition from boundary-pure.cpp
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

    // =========================================================================
    // YELLOW ZONE: DOWN BOUNDARY CONDITION - AFTER STEP (l > L_end, m = 0)
    // Same condition as boundary-pure.cpp - characteristic-based BC
    // =========================================================================
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

    // =========================================================================
    // RIGHT BOUNDARY CONDITION (z = z_max) - OUTFLOW
    // Same as boundary-pure.cpp - Only for last rank
    // =========================================================================
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

