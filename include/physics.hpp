#pragma once

// Compute maximum wave speed for CFL condition
double compute_max_wave_speed(double **rho, double **v_z, double **v_r, double **v_phi,
                               double **H_z, double **H_r, double **H_phi, double **p,
                               int local_L, int M_max, double gamma);

// Compute relative solution change for convergence checking
double compute_solution_change(double **rho_curr, double **rho_prev,
                                double **v_z_curr, double **v_z_prev,
                                double **v_r_curr, double **v_r_prev,
                                double **v_phi_curr, double **v_phi_prev,
                                double **H_z_curr, double **H_z_prev,
                                double **H_r_curr, double **H_r_prev,
                                double **H_phi_curr, double **H_phi_prev,
                                int local_L, int M_max);

// Helper function to find maximum in array
double max_array(double **array, int L, int M);
