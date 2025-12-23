#include "integrals.hpp"

#include <cmath>

auto TrapezoidIntegrate(double *func, double dr, int n_points) -> double {

    double result = 0;

    for (int i = 0; i < n_points - 1; i++) {
        result += func[i] * i + func[i + 1] * (i + 1);
    }

    result = result * dr * dr * M_PI;

    return result;
}

auto GetMassFlux(double *rho_last_col, double *v_z_last_col,
                     double dr_last_col, int M_max) -> double {

    auto *rho_times_v_z = new double[M_max + 1];

    for (int i = 0; i < M_max + 1; i++)
        rho_times_v_z[i] = rho_last_col[i] * v_z_last_col[i];

    double mass_flux = TrapezoidIntegrate(rho_times_v_z, dr_last_col, M_max);
    delete[] rho_times_v_z;

    return mass_flux;
}

auto GetThrust(double *rho_last_col, double *v_z_last_col,
                  double *p_last_col, double *H_r_last_col,
                  double *H_phi_last_col, double *H_z_last_col,
                  double dr_last_col, int M_max) -> double {

    auto *stress_tensor = new double[M_max + 1];

    for (int i = 0; i < M_max + 1; i++) {
        double v2 = v_z_last_col[i] * v_z_last_col[i];
        double H2 = H_r_last_col[i] * H_r_last_col[i] +
                    H_phi_last_col[i] * H_phi_last_col[i] +
                    H_z_last_col[i] * H_z_last_col[i];
        stress_tensor[i] =
            rho_last_col[i] * v2 + p_last_col[i] + H2 / (8.0 * M_PI);
    }

    double thrust = TrapezoidIntegrate(stress_tensor, dr_last_col, M_max);
    delete[] stress_tensor;

    return thrust;
}
