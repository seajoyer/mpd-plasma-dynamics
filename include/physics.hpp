#pragma once

#include "types.hpp"

// Compute maximum wave speed for CFL condition
auto ComputeMaxWaveSpeed(double **rho, double **v_z, double **v_r, double **v_phi,
                               double **H_z, double **H_r, double **H_phi, double **p,
                               int local_L, int M_max, double gamma) -> double;

// Compute relative solution change for convergence checking
auto ComputeSolutionChange(double **rho_curr, double **rho_prev,
                                double **v_z_curr, double **v_z_prev,
                                double **v_r_curr, double **v_r_prev,
                                double **v_phi_curr, double **v_phi_prev,
                                double **H_z_curr, double **H_z_prev,
                                double **H_r_curr, double **H_r_prev,
                                double **H_phi_curr, double **H_phi_prev,
                                int local_L, int M_max) -> double;

// Helper function to find maximum in array
auto MaxArray(double **array, int L, int M) -> double;

// Compute adaptive time step based on CFL condition
// Returns the new dt value that satisfies CFL < CFL_target
// Takes into account variable cell sizes (dz and dr[l])
auto ComputeAdaptiveTimeStep(const PhysicalFields& fields, const GridGeometry& grid,
                              const DomainInfo& domain, const SimulationParams& params) -> double;
