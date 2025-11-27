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
    
    // Global l indices for step transition region
    const int l_step_start = static_cast<int>(z_start / dz);
    const int l_step_end = static_cast<int>(z_end / dz);
    
    // Calculate number of m-cells that span the lateral gap
    // At z > z_end: r(l,m) = (1-m*dy)*R1_after + m*dy*R2
    // The gap is from r = r_after (0.005) to r = r_before (0.2)
    // m_gap where r(l, m_gap) = r_before
    const double dy = params.dy;
    const double R2_val = 0.8;  // Outer radius (constant)
    // r = r_after + m * dy * (R2 - r_after) = r_before
    // m_gap = (r_before - r_after) / (dy * (R2 - r_after))
    const int m_gap = static_cast<int>((r_before - r_after) / (dy * (R2_val - r_after)));
    
    // Inflow velocity through lateral gap (can be adjusted)
    const double v_inflow = 0.1;  // Axial velocity component entering the channel
    
    // =========================================================================
    // LEFT BOUNDARY CONDITION (z=0) - WALL / NO-FLOW
    // Only for rank 0
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
    // Outflow / extrapolation boundary
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
    // DOWN BOUNDARY CONDITION - BEFORE STEP (l < l_step_start)
    // Wall / no-flow at inner solid boundary
    // =========================================================================
    if (domain.l_start < l_step_start && domain.l_end >= 1) {
        int L_wall_end = std::min(l_step_start - 1, domain.l_end);
        int local_L_wall_end = L_wall_end - domain.l_start + 1;
        
        for (int l = 1; l <= local_L_wall_end; l++) {
            int l_global = domain.l_start + l - 1;
            if (l_global >= 1 && l_global < l_step_start) {
                // Wall boundary (no-flow) at inner surface before the step
                fields.rho[l][0] = fields.rho[l][1];
                fields.v_z[l][0] = fields.v_z[l][1];
                fields.v_r[l][0] = 0.0;  // No flow through wall
                fields.v_phi[l][0] = fields.v_phi[l][1];
                fields.e[l][0] = fields.e[l][1];
                fields.H_phi[l][0] = fields.H_phi[l][1];
                fields.H_z[l][0] = fields.H_z[l][1];
                fields.H_r[l][0] = 0.0;  // No normal B-field at wall

                u.u_1[l][0] = fields.rho[l][0] * grid.r[l][0];
                u.u_2[l][0] = fields.rho[l][0] * fields.v_z[l][0] * grid.r[l][0];
                u.u_3[l][0] = 0.0;  // No radial momentum flux
                u.u_4[l][0] = fields.rho[l][0] * fields.v_phi[l][0] * grid.r[l][0];
                u.u_5[l][0] = fields.rho[l][0] * fields.e[l][0] * grid.r[l][0];
                u.u_6[l][0] = fields.H_phi[l][0];
                u.u_7[l][0] = fields.H_z[l][0] * grid.r[l][0];
                u.u_8[l][0] = 0.0;  // No radial B-field flux
            }
        }
    }

    // =========================================================================
    // LATERAL STEP INFLOW BOUNDARY (l_step_start <= l <= l_step_end)
    // This is the lateral gap where gas enters from the side surface of the step
    // Gas flows through the gap between r_after and r_before at the step
    // =========================================================================
    if (domain.l_start <= l_step_end && domain.l_end >= l_step_start) {
        int local_l_start = std::max(1, l_step_start - domain.l_start + 1);
        int local_l_end = std::min(local_L, l_step_end - domain.l_start + 1);
        
        for (int l = local_l_start; l <= local_l_end; l++) {
            int l_global = domain.l_start + l - 1;
            if (l_global >= l_step_start && l_global <= l_step_end) {
                // Calculate local step geometry progress (0 to 1 through transition)
                double z_local = l_global * dz;
                double xi = (z_local - z_start) / (z_end - z_start);
                xi = std::max(0.0, std::min(1.0, xi));  // Clamp to [0,1]
                
                // Smooth transition factor (same as in geometry.cpp)
                double smooth_factor = 0.5 * (1.0 - cos(M_PI * xi));
                
                // Current R1 at this z-location
                double R1_local = r_before + (r_after - r_before) * smooth_factor;
                
                // Calculate how many m-cells are in the "gap" region at this l
                // The gap extends from R1_local up to r_before
                int m_gap_local = static_cast<int>((r_before - R1_local) / grid.dr[l]);
                m_gap_local = std::max(1, std::min(m_gap_local, m_gap));
                
                // Apply inflow conditions at lower m values (the gap region)
                for (int m = 0; m <= m_gap_local; m++) {
                    // Inflow through lateral gap
                    // Gas enters with axial velocity component (into the channel)
                    // and radial velocity component (inward toward axis)
                    
                    fields.rho[l][m] = 1.0;  // Inlet density
                    fields.v_phi[l][m] = 0.0;
                    
                    // Velocity: primarily axial (into the nozzle), with radial component
                    // The radial component is inward (negative) to simulate flow from the gap
                    double radial_factor = smooth_factor;  // More radial at end of transition
                    fields.v_z[l][m] = v_inflow * (1.0 - 0.3 * radial_factor);
                    fields.v_r[l][m] = -v_inflow * 0.5 * radial_factor;  // Inward (negative r direction)
                    
                    // Magnetic field at inlet
                    fields.H_phi[l][m] = r_0 / grid.r[l][m];
                    fields.H_z[l][m] = H_z0;
                    fields.H_r[l][m] = fields.H_z[l][m] * grid.r_z[l][m];
                    
                    // Internal energy
                    fields.e[l][m] = beta / (2.0 * (gamma - 1.0)) * pow(fields.rho[l][m], gamma - 1.0);

                    // Update conservative variables
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
        }
    }

    // =========================================================================
    // DOWN BOUNDARY CONDITION - AFTER STEP TRANSITION (l_step_end < l <= L_end)
    // Inflow region in the narrow channel near the step
    // =========================================================================
    if (domain.l_start <= L_end && domain.l_end > l_step_end) {
        int local_l_start = std::max(1, l_step_end + 1 - domain.l_start + 1);
        int local_l_end = std::min(local_L, L_end - domain.l_start + 1);
        
        for (int l = local_l_start; l <= local_l_end; l++) {
            int l_global = domain.l_start + l - 1;
            if (l_global > l_step_end && l_global <= L_end) {
                // In the narrow channel after the step, apply inflow at lower boundary
                // This represents flow that has entered through the lateral gap
                
                // Apply inflow at cells in the gap region (m from 0 to m_gap)
                int m_inflow_max = std::min(m_gap, M_max - 1);
                
                for (int m = 0; m <= m_inflow_max; m++) {
                    // Distance from step (for smooth transition)
                    double dist_factor = static_cast<double>(l_global - l_step_end) / 
                                        static_cast<double>(L_end - l_step_end);
                    dist_factor = std::min(1.0, dist_factor);
                    
                    // Radial position factor (less inflow intensity at higher m)
                    double m_factor = 1.0 - static_cast<double>(m) / static_cast<double>(m_inflow_max + 1);
                    
                    fields.rho[l][m] = 1.0;
                    fields.v_phi[l][m] = 0.0;
                    
                    // Velocity: primarily axial, decreasing radial component with distance from step
                    fields.v_z[l][m] = v_inflow;
                    fields.v_r[l][m] = -v_inflow * 0.3 * (1.0 - dist_factor) * m_factor;  // Decreasing inward flow
                    
                    // Magnetic field at inlet
                    fields.H_phi[l][m] = r_0 / grid.r[l][m];
                    fields.H_z[l][m] = H_z0;
                    fields.H_r[l][m] = fields.H_z[l][m] * grid.r_z[l][m];
                    
                    // Internal energy
                    fields.e[l][m] = beta / (2.0 * (gamma - 1.0)) * pow(fields.rho[l][m], gamma - 1.0);

                    // Update conservative variables
                    u.u_1[l][m] = fields.rho[l][m] * grid.r[l][m];
                    u.u_2[l][m] = fields.rho[l][m] * fields.v_z[l][m] * grid.r[l][m];
                    u.u_3[l][m] = fields.rho[l][m] * fields.v_r[l][m] * grid.r[l][m];
                    u.u_4[l][m] = fields.rho[l][m] * fields.v_phi[l][m] * grid.r[l][m];
                    u.u_5[l][m] = fields.rho[l][m] * fields.e[l][m] * grid.r[l][m];
                    u.u_6[l][m] = fields.H_phi[l][m];
                    u.u_7[l][m] = fields.H_z[l][m] * grid.r[l][m];
                    u.u_8[l][m] = fields.H_r[l][m] * grid.r[l][m];
                }
                
                // For m > m_inflow_max (above the gap), apply wall BC at m=0 only
                if (m_inflow_max < M_max) {
                    // The cells above the gap region at the inner boundary
                    // These are at r > r_before and should not have special treatment
                    // (they are interior cells, not at the actual inner boundary)
                }
            }
        }
    }

    // =========================================================================
    // DOWN BOUNDARY CONDITION - FAR FROM STEP (l > L_end)
    // Open/outflow boundary condition
    // =========================================================================
    #pragma omp parallel for
    for (int l = 1; l < local_L + 1; l++) {
        int l_global = domain.l_start + l - 1;
        
        if (l_global > L_end && l_global < L_max_global) {
            int m = 0;
            
            // Characteristic-based outflow boundary
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
    // Only for last rank
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
