#include "physics.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

auto ComputeMaxWaveSpeed(double **rho, double **v_z, double **v_r, double **v_phi,
                               double **H_z, double **H_r, double **H_phi, double **p,
                               int local_L, int M_max, double gamma) -> double {
    double max_speed = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(max:max_speed)
    for (int l = 1; l < local_L + 1; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            // Sound speed
            double cs = std::sqrt(gamma * p[l][m] / rho[l][m]);
            
            // Alfven speed (fast magnetosonic)
            double ca = std::sqrt((H_z[l][m]*H_z[l][m] + H_r[l][m]*H_r[l][m] + 
                                   H_phi[l][m]*H_phi[l][m]) / rho[l][m]);
            
            // Flow speed
            double v = std::sqrt(v_z[l][m]*v_z[l][m] + v_r[l][m]*v_r[l][m]);
            
            // Maximum characteristic speed (fast magnetosonic wave)
            double local_speed = v + cs + ca;
            max_speed = std::max(max_speed, local_speed);
        }
    }
    
    // Find global maximum across all processes
    double global_max_speed;
    MPI_Allreduce(&max_speed, &global_max_speed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    return global_max_speed;
}

auto ComputeSolutionChange(double **rho_curr, double **rho_prev,
                                double **v_z_curr, double **v_z_prev,
                                double **v_r_curr, double **v_r_prev,
                                double **v_phi_curr, double **v_phi_prev,
                                double **H_z_curr, double **H_z_prev,
                                double **H_r_curr, double **H_r_prev,
                                double **H_phi_curr, double **H_phi_prev,
                                int local_L, int M_max) -> double {

    double sum_sq_diff = 0.0;
    double sum_sq_curr = 0.0;

    // OpenMP parallel reduction over interior cells (excluding ghost cells)
    #pragma omp parallel for collapse(2) reduction(+:sum_sq_diff, sum_sq_curr)
    for (int l = 1; l < local_L + 1; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            // Density contribution
            double d_rho = rho_curr[l][m] - rho_prev[l][m];
            sum_sq_diff += d_rho * d_rho;
            sum_sq_curr += rho_curr[l][m] * rho_curr[l][m];

            // Velocity contributions
            double d_vz = v_z_curr[l][m] - v_z_prev[l][m];
            double d_vr = v_r_curr[l][m] - v_r_prev[l][m];
            double d_vphi = v_phi_curr[l][m] - v_phi_prev[l][m];
            sum_sq_diff += d_vz * d_vz + d_vr * d_vr + d_vphi * d_vphi;
            sum_sq_curr += v_z_curr[l][m] * v_z_curr[l][m] +
                           v_r_curr[l][m] * v_r_curr[l][m] +
                           v_phi_curr[l][m] * v_phi_curr[l][m];

            // Magnetic field contributions
            double d_Hz = H_z_curr[l][m] - H_z_prev[l][m];
            double d_Hr = H_r_curr[l][m] - H_r_prev[l][m];
            double d_Hphi = H_phi_curr[l][m] - H_phi_prev[l][m];
            sum_sq_diff += d_Hz * d_Hz + d_Hr * d_Hr + d_Hphi * d_Hphi;
            sum_sq_curr += H_z_curr[l][m] * H_z_curr[l][m] +
                           H_r_curr[l][m] * H_r_curr[l][m] +
                           H_phi_curr[l][m] * H_phi_curr[l][m];
        }
    }

    // MPI reduction across all processes
    double global_sum_sq_diff = 0.0;
    double global_sum_sq_curr = 0.0;
    MPI_Allreduce(&sum_sq_diff, &global_sum_sq_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_curr, &global_sum_sq_curr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double norm_diff = std::sqrt(global_sum_sq_diff);
    double norm_curr = std::sqrt(global_sum_sq_curr);

    if (norm_curr > 1e-15) {
        return norm_diff / norm_curr;
    }

    return norm_diff;
}

auto MaxArray(double **array, int L, int M) -> double {
    double maxim = 0.0;

    for (int l = 0; l < L + 1; l++) {
        for (int m = 0; m < M + 1; m++) {
            if (maxim < array[l][m]) {
                maxim = array[l][m];
            }
        }
    }

    return maxim;
}
