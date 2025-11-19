#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdio.h>

/**
 * @brief Simple convergence check function
 *
 * Computes a combined relative L² norm across all key variables.
 *
 * @return Combined relative change in solution
 */
double compute_solution_change(double **rho_curr, double **rho_prev,
                               double **v_z_curr, double **v_z_prev,
                               double **v_r_curr, double **v_r_prev,
                               double **v_phi_curr, double **v_phi_prev,
                               double **H_z_curr, double **H_z_prev,
                               double **H_r_curr, double **H_r_prev,
                               double **H_phi_curr, double **H_phi_prev,
                               int L_max, int M_max) {

    double sum_sq_diff = 0.0;
    double sum_sq_curr = 0.0;

#pragma omp parallel for collapse(2) reduction(+ : sum_sq_diff, sum_sq_curr)
    for (int l = 0; l < L_max + 1; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            // Density contribution
            double d_rho = rho_curr[l][m] - rho_prev[l][m];
            sum_sq_diff += d_rho * d_rho;
            sum_sq_curr += rho_curr[l][m] * rho_curr[l][m];

            // Velocity contributions (weighted)
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

    double norm_diff = std::sqrt(sum_sq_diff);
    double norm_curr = std::sqrt(sum_sq_curr);

    if (norm_curr > 1e-15) {
        return norm_diff / norm_curr;
    }

    return norm_diff;
}

/**
 * Allocates memory for a 2D double array and initializes all elements to zero
 * @param array Reference to the array pointer to be allocated
 * @param rows Number of rows in the array
 * @param columns Number of columns in the array
 */
void memory_allocation_2D(double **&array, int rows, int columns) {
    array = new double *[rows];
    for (int i = 0; i < rows; i++) {
        array[i] = new double[columns];
        for (int j = 0; j < columns; j++) {
            array[i][j] = 0;
        }
    }
}

/**
 * Deallocates memory for a 2D double array
 * @param array Reference to the array pointer to be deallocated
 * @param rows Number of rows in the array
 */
void memory_clearing_2D(double **&array, int rows) {
    for (int i = 0; i < rows; i++) {
        delete[] array[i];
    }
    delete[] array;
}

/**
 * Inner boundary function r1(z) - defines the inner radius as a function of z
 * This creates a variable geometry with a constriction
 * @param z Axial coordinate
 * @return Inner radius value
 */
double r1(double z) {
    const double z_center = 0.31;      // Центр ступеньки
    const double transition_width = 0.015; // Полуширина перехода (уже в ~5 раз)
    const double r_before = 0.2;       // Радиус до ступеньки
    const double r_after = 0.005;      // Радиус после ступеньки (узкое горло)

    // Начало и конец переходной зоны
    const double z_start = z_center - transition_width;
    const double z_end = z_center + transition_width;

    if (z < z_start) {
        // Постоянный радиус до ступеньки
        return r_before;
    }
    else if (z >= z_start && z < z_end) {
        // Плавный переход используя косинусоидальную функцию (C² непрерывность)
        // Это обеспечивает гладкость производных
        double xi = (z - z_start) / (z_end - z_start); // нормализованная координата [0,1]
        double smooth_factor = 0.5 * (1.0 - cos(M_PI * xi)); // S-образная кривая
        return r_before + (r_after - r_before) * smooth_factor;
    }
    else {
        // Узкое горло после ступеньки
        return r_after;
    }
}

/**
 * Outer boundary function r2(z) - defines the outer radius as a function of z
 * @param z Axial coordinate
 * @return Outer radius value (constant)
 */
double r2(double z) {
    return 0.8; // Constant outer radius
}

/**
 * Derivative of inner boundary function dr1/dz
 * @param z Axial coordinate
 * @return Derivative of inner radius
 */
double der_r1(double z) {
    const double z_center = 0.31;
    const double transition_width = 0.015;
    const double r_before = 0.2;
    const double r_after = 0.005;

    const double z_start = z_center - transition_width;
    const double z_end = z_center + transition_width;

    if (z < z_start) {
        return 0.0; // Постоянный радиус - нулевая производная
    }
    else if (z >= z_start && z < z_end) {
        // Производная косинусоидальной функции
        double xi = (z - z_start) / (z_end - z_start);
        double dxi_dz = 1.0 / (z_end - z_start);
        double dsmooth_dxi = 0.5 * M_PI * sin(M_PI * xi);
        return (r_after - r_before) * dsmooth_dxi * dxi_dz;
    }
    else {
        return 0.0; // Постоянный радиус после ступеньки
    }
}

/**
 * Derivative of outer boundary function dr2/dz
 * @param z Axial coordinate
 * @return Derivative of outer radius (always zero for constant boundary)
 */
double der_r2(double z) {
    return 0; // Constant outer boundary has zero derivative
}

/**
 * Check if a given z-coordinate is in the inlet region (step surface)
 * @param z Axial coordinate
 * @return true if in inlet region
 */
bool is_inlet_region(double z) {
    const double z_center = 0.31;
    const double transition_width = 0.015;
    const double z_start = z_center - transition_width;
    const double z_end = z_center + transition_width;
    
    // Inlet occurs in the transition zone where the step is located
    return (z >= z_start && z <= z_end);
}

/**
 * @brief Numerically integrates a radially-symmetric function over a 2D
 * circular domain using the trapezoidal rule in polar coordinates.
 *
 * This function computes the integral of a function f(r) over a full circle (0
 * to 2π in angle) by applying the trapezoidal rule to the radial component. The
 * integration assumes the function is provided as discrete samples at equally
 * spaced radial points starting from r = 0.
 *
 * The integral computed is:
 * \f[
 *   \int_0^{2\pi} \int_0^{r_{\text{max}}} f(r) \, r \, dr \, d\theta = 2\pi
 * \int_0^{r_{\text{max}}} f(r) \, r \, dr
 * \f]
 *
 * where \f$ r_{\text{max}} = (n\_points - 1) \cdot dr \f$.
 *
 * @param func      Pointer to an array of function values f(r) sampled at
 * radial points. The array must contain at least `n_points` elements, where
 *                      `func[i]` corresponds to f(i * dr).
 * @param dr        Radial step size (spacing between consecutive sample
 * points).
 * @param n_points  Number of sample points in the `func` array. Must be >= 2.
 *
 * @return The approximate value of the 2D integral over the circular domain.
 *
 * @note The function assumes the input samples start at r = 0 and are uniformly
 * spaced by `dr`.
 * @note This implementation includes the Jacobian factor `r` from the polar
 * coordinate transformation and the full angular integration factor `2π`.
 *
 * @warning Behavior is undefined if `n_points < 2` or if `dr <= 0`.
 */
double trapezoid_integrate(double *func, double dr, int n_points) {

    double result = 0;

    for (int i = 0; i < n_points - 1; i++)
        result += func[i] * i + func[i + 1] * (i + 1);

    result = result * dr * dr * M_PI;

    return result;
}

/**
 * Calculate the mass flux thoughout the right boundary of the nozzle
 * @param rho_last_col density on right boundary
 * @param v_z_last_col velocity on right boundary
 * @param dr_last_col  axial grid spacind on right boundary
 * @param M_max        number of points on the right boundaty
 * @return Mass flux thoughout the right boundary of the nozzle
 */
double get_mass_flux(double *rho_last_col, double *v_z_last_col,
                     double dr_last_col, int M_max) {

    double *rho_times_v_z = new double[M_max + 1];

    for (int i = 0; i < M_max + 1; i++)
        rho_times_v_z[i] = rho_last_col[i] * v_z_last_col[i];

    double mass_flux = trapezoid_integrate(rho_times_v_z, dr_last_col, M_max);
    delete[] rho_times_v_z;

    return mass_flux;
}

/**
 * @brief Computes the axial thrust using the stress tensor integrated over the
 * last radial column.
 *
 * This function calculates the thrust by evaluating the axial component of the
 * stress tensor at each radial grid point in the last column of the
 * computational domain and then integrating it using the trapezoidal rule
 *
 * @param rho_last_col      Array of density values at the last radial column
 * (size: M_max + 1).
 * @param v_z_last_col      Array of axial velocity components at the last
 * radial column (size: M_max + 1).
 * @param p_last_col        Array of gas pressure values at the last radial
 * column (size: M_max + 1).
 * @param H_r_last_col      Array of radial magnetic field components at the
 * last radial column (size: M_max + 1).
 * @param H_phi_last_col    Array of azimuthal magnetic field components at the
 * last radial column (size: M_max + 1).
 * @param H_z_last_col      Array of axial magnetic field components at the last
 * radial column (size: M_max + 1).
 * @param dr_last_col       Radial grid spacing at the last column (assumed
 * uniform).
 * @param M_max             Maximum radial index (number of radial zones minus
 * one).
 *
 * @return The integrated axial thrust value.
 */
double get_thrust(double *rho_last_col, double *v_z_last_col,
                  double *p_last_col, double *H_r_last_col,
                  double *H_phi_last_col, double *H_z_last_col,
                  double dr_last_col, int M_max) {

    double *stress_tensor = new double[M_max + 1];

    for (int i = 0; i < M_max + 1; i++) {
        double v2 = v_z_last_col[i] * v_z_last_col[i];
        double H2 = H_r_last_col[i] * H_r_last_col[i] +
                    H_phi_last_col[i] * H_phi_last_col[i] +
                    H_z_last_col[i] * H_z_last_col[i];
        stress_tensor[i] =
            rho_last_col[i] * v2 + p_last_col[i] + H2 / (8.0 * M_PI);
    }

    double thrust = trapezoid_integrate(stress_tensor, dr_last_col, M_max);
    delete[] stress_tensor;

    return thrust;
}

/**
 * Writes simulation data to file for animation/visualization
 * Outputs data in Tecplot format for post-processing
 * @param n Time step number (for filename)
 * @param L_max Maximum number of axial grid points
 * @param M_max Maximum number of radial grid points
 * @param dz Axial grid spacing
 * @param r Radial coordinate array
 * @param rho Density array
 * @param v_z Axial velocity array
 * @param v_r Radial velocity array
 * @param v_phi Azimuthal velocity array
 * @param e Energy density array
 * @param H_z Axial magnetic field array
 * @param H_r Radial magnetic field array
 * @param H_phi Azimuthal magnetic field array
 */
void animate_write(int n, int L_max, int M_max, double dz, double **r,
                   double **rho, double **v_z, double **v_r, double **v_phi,
                   double **e, double **H_z, double **H_r, double **H_phi) {

    // Generate filename based on time step
    char filename[50];
    sprintf(filename, "animate_m_29_800x400/%d.plt", n);

    std::ofstream out(filename);
    int np = (L_max + 1) * (M_max + 1); // Total number of points
    int ne = L_max * M_max;             // Total number of elements
    double hfr;                         // H_phi * r for output

    // Write Tecplot header
    out << "VARIABLES=\n";
    out << "\"z\"\n\"r\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n\"Energy\""
           "\n"
           "\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
    out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne
        << "\n ";

    // Write point data
    for (int m = 0; m < M_max + 1; m++) {
        for (int l = 0; l < L_max + 1; l++) {
            hfr = H_phi[l][m] * r[l][m];
            out << l * dz << " " << r[l][m] << " " << rho[l][m] << " "
                << v_z[l][m] << " " << v_r[l][m] << " "
                << std::sqrt(v_z[l][m] * v_z[l][m] + v_r[l][m] * v_r[l][m])
                << " " << v_phi[l][m] << " " << e[l][m] << " " << H_z[l][m]
                << " " << H_r[l][m] << " " << hfr << " " << H_phi[l][m] << "\n";
        }
    }

    // Write element connectivity (quadrilateral elements)
    int i1 = 0, i2 = 0, i3 = 0, i4 = 0;
    for (int m = 0; m < M_max; m++) {
        for (int l = 0; l < L_max; l++) {
            i1 = l + m * (L_max + 1) + 1;
            i2 = l + 1 + m * (L_max + 1) + 1;
            i3 = l + 1 + (m + 1) * (L_max + 1) + 1;
            i4 = l + (m + 1) * (L_max + 1) + 1;
            out << i1 << " " << i2 << " " << i3 << " " << i4 << "\n";
        }
    }

    out.close();
}

/**
 * Writes simulation data to VTK file for Paraview visualization
 * @param filename Output filename
 * @param L_max Maximum number of axial grid points
 * @param M_max Maximum number of radial grid points
 * @param dz Axial grid spacing
 * @param r Radial coordinate array
 * @param rho Density array
 * @param v_z Axial velocity array
 * @param v_r Radial velocity array
 * @param v_phi Azimuthal velocity array
 * @param e Energy density array
 * @param H_z Axial magnetic field array
 * @param H_r Radial magnetic field array
 * @param H_phi Azimuthal magnetic field array
 */
void write_vtk(const char *filename, int L_max, int M_max, double dz,
               double **r, double **rho, double **v_z, double **v_r,
               double **v_phi, double **e, double **H_z, double **H_r,
               double **H_phi) {

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening " << filename << std::endl;
        return;
    }

    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "MHD Simulation" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET STRUCTURED_GRID" << std::endl;

    int nx = L_max + 1; // z direction (fast varying)
    int ny = M_max + 1; // r direction (slow varying)
    int nz = 1;
    out << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;

    int npoints = nx * ny * nz;
    out << "POINTS " << npoints << " float" << std::endl;

    // Write points: r as x, z as y, 0 as z_coord
    for (int j = 0; j < ny; ++j) {     // m (r)
        for (int i = 0; i < nx; ++i) { // l (z)
            double x = r[i][j];        // r[l][m]
            double y = i * dz;         // z = l * dz
            double z_coord = 0.0;
            out << x << " " << y << " " << z_coord << std::endl;
        }
    }

    out << "POINT_DATA " << npoints << std::endl;

    // SCALARS Rho
    out << "SCALARS Rho float 1" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            out << rho[i][j] << std::endl;
        }
    }

    // VECTORS Velocity (v_r, v_z, 0)
    out << "VECTORS Velocity float" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            out << v_r[i][j] << " " << v_z[i][j] << " 0.0" << std::endl;
        }
    }

    // SCALARS Vphi
    out << "SCALARS Vphi float 1" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            out << v_phi[i][j] << std::endl;
        }
    }

    // SCALARS Energy
    out << "SCALARS Energy float 1" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            out << e[i][j] << std::endl;
        }
    }

    // VECTORS MagneticField (H_r, H_z, 0)
    out << "VECTORS MagneticField float" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            out << H_r[i][j] << " " << H_z[i][j] << " 0.0" << std::endl;
        }
    }

    // SCALARS Hphi
    out << "SCALARS Hphi float 1" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            out << H_phi[i][j] << std::endl;
        }
    }

    // SCALARS Vl = sqrt(v_z^2 + v_r^2)
    out << "SCALARS Vl float 1" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double vl =
                std::sqrt(v_z[i][j] * v_z[i][j] + v_r[i][j] * v_r[i][j]);
            out << vl << std::endl;
        }
    }

    // SCALARS Hphi*r
    out << "SCALARS Hphi_r float 1" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            out << H_phi[i][j] * r[i][j] << std::endl;
        }
    }

    out.close();
}

/**
 * Finds the maximum value in a 2D array
 * Used for stability analysis and time step control
 * @param array 2D array to search
 * @param L Number of rows to search
 * @param M Number of columns to search
 * @return Maximum value found in the array
 */
double max_array(double **array, double L, double M) {
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

/**
 * Displays a progress bar in the console
 * @param current Current iteration number
 * @param total Total number of iterations
 * @param current_time Current simulation time
 * @param total_time Total simulation time
 */
void show_progress(int current, int total, double current_time,
                   double total_time) {
    const int bar_width = 50;
    double progress = (double)current / total;

    std::cout << "\r[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100.0
              << "% | Time: " << std::setprecision(4) << current_time << "/"
              << total_time;
    std::cout.flush();
}

int main(int argc, char *argv[]) {

    // ============================================================================
    // PHYSICAL PARAMETERS AND SIMULATION SETUP
    // ============================================================================

    double gamma = 1.67; // Adiabatic index (ratio of specific heats)
    double beta =
        0.05; // Plasma beta parameter (thermal to magnetic pressure ratio)
    double H_z0 = 0.0; // Initial axial magnetic field strength

    int animate = 0; // Animation flag (0 = off, 1 = on)

    // Inlet parameters (for step surface injection)
    double inlet_velocity = 0.5;  // Radial inlet velocity magnitude
    double inlet_density = 1.0;   // Inlet density
    
    // ============================================================================
    // COMPUTATIONAL DOMAIN AND TIME DISCRETIZATION
    // ============================================================================

    double T = 10.0;      // Total simulation time
    double t = 0.0;       // Current time
    double dt = 0.0000125; // Time step size

    int L_max = 800; // Number of axial grid points
    int L_end = 265; // Transition point for boundary conditions
    int M_max = 400; // Number of radial grid points

    double dz = 1.0 / L_max; // Axial grid spacing
    double dy = 1.0 / M_max; // Normalized radial coordinate spacing

    // ============================================================================
    // PARALLEL COMPUTING SETUP
    // ============================================================================

    int procs = atoi(argv[1]); // Number of threads from command line
    omp_set_num_threads(procs);

    double begin, end, total; // Timing variables

    // ============================================================================
    // MEMORY ALLOCATION FOR CONSERVATIVE VARIABLES
    // ============================================================================

    // Conservative variables u = {rho*r*R, rho*r*v_z*R, rho*r*v_r*R,
    //                            rho*r*v_phi*R, rho*r*e*R, H_phi*R, H_z*r*R,
    //                            H_y*r}
    // where R is the local domain width

    // Current time step conservative variables
    double **u_1, **u_2, **u_3, **u_4, **u_5, **u_6, **u_7, **u_8;
    memory_allocation_2D(u_1, L_max + 1, M_max + 1); // Mass density
    memory_allocation_2D(u_2, L_max + 1, M_max + 1); // Axial momentum
    memory_allocation_2D(u_3, L_max + 1, M_max + 1); // Radial momentum
    memory_allocation_2D(u_4, L_max + 1, M_max + 1); // Azimuthal momentum
    memory_allocation_2D(u_5, L_max + 1, M_max + 1); // Energy density
    memory_allocation_2D(u_6, L_max + 1, M_max + 1); // Azimuthal magnetic field
    memory_allocation_2D(u_7, L_max + 1, M_max + 1); // Axial magnetic field
    memory_allocation_2D(u_8, L_max + 1, M_max + 1); // Radial magnetic field

    // Previous time step conservative variables (for time integration)
    double **u0_1, **u0_2, **u0_3, **u0_4, **u0_5, **u0_6, **u0_7, **u0_8;
    memory_allocation_2D(u0_1, L_max + 1, M_max + 1);
    memory_allocation_2D(u0_2, L_max + 1, M_max + 1);
    memory_allocation_2D(u0_3, L_max + 1, M_max + 1);
    memory_allocation_2D(u0_4, L_max + 1, M_max + 1);
    memory_allocation_2D(u0_5, L_max + 1, M_max + 1);
    memory_allocation_2D(u0_6, L_max + 1, M_max + 1);
    memory_allocation_2D(u0_7, L_max + 1, M_max + 1);
    memory_allocation_2D(u0_8, L_max + 1, M_max + 1);

    // ============================================================================
    // MEMORY ALLOCATION FOR PRIMITIVE VARIABLES
    // ============================================================================

    double **rho;   // Mass density
    double **v_r;   // Radial velocity
    double **v_phi; // Azimuthal velocity
    double **v_z;   // Axial velocity
    double **e;     // Specific internal energy
    double **p;     // Thermal pressure
    double **P;     // Total pressure (thermal + magnetic)
    double **H_r;   // Radial magnetic field
    double **H_phi; // Azimuthal magnetic field
    double **H_z;   // Axial magnetic field

    memory_allocation_2D(rho, L_max + 1, M_max + 1);
    memory_allocation_2D(v_r, L_max + 1, M_max + 1);
    memory_allocation_2D(v_phi, L_max + 1, M_max + 1);
    memory_allocation_2D(v_z, L_max + 1, M_max + 1);
    memory_allocation_2D(e, L_max + 1, M_max + 1);
    memory_allocation_2D(p, L_max + 1, M_max + 1);
    memory_allocation_2D(P, L_max + 1, M_max + 1);
    memory_allocation_2D(H_r, L_max + 1, M_max + 1);
    memory_allocation_2D(H_phi, L_max + 1, M_max + 1);
    memory_allocation_2D(H_z, L_max + 1, M_max + 1);

    // ============================================================================
    // MEMORY ALLOCATION FOR GEOMETRIC VARIABLES
    // ============================================================================

    double **r;   // Radial coordinate at each grid point
    double **r_z; // Derivative dr/dz at each grid point
    memory_allocation_2D(r, L_max + 1, M_max + 1);
    memory_allocation_2D(r_z, L_max + 1, M_max + 1);

    double *R =
        new double[L_max + 1]; // Local domain width R(z) = r2(z) - r1(z)
    double *dr = new double[L_max + 1]; // Local radial grid spacing

    // Initialize geometric arrays
    for (int l = 0; l < L_max + 1; l++) {
        R[l] = 0;
        dr[l] = 0;
    }

    // ============================================================================
    // GRID GENERATION
    // ============================================================================

    double r_0 = (r1(0) + r2(0)) / 2.0; // Reference radius at z=0

    // Generate structured grid in the variable geometry domain
    for (int l = 0; l < L_max + 1; l++) {
        R[l] = r2(l * dz) - r1(l * dz); // Local domain width
        dr[l] = R[l] / M_max;           // Local radial spacing

        for (int m = 0; m < M_max + 1; m++) {
            // Linear interpolation between inner and outer boundaries
            r[l][m] = (1 - m * dy) * r1(l * dz) + m * dy * r2(l * dz);
            // Derivative of radius with respect to z
            r_z[l][m] = (1 - m * dy) * der_r1(l * dz) + m * dy * der_r2(l * dz);
        }
    }

    // ============================================================================
    // CONVERGENCE MONITORING SETUP
    // ============================================================================

    double convergence_tolerance = 1e-3;  // Adjust as needed
    int convergence_check_interval = 500; // Check every N steps
    bool enable_convergence_check = true; // Set to false to disable

    // Allocate arrays to store previous solution for comparison
    double **rho_check, **v_z_check, **v_r_check, **v_phi_check;
    double **H_z_check, **H_r_check, **H_phi_check;

    memory_allocation_2D(rho_check, L_max + 1, M_max + 1);
    memory_allocation_2D(v_z_check, L_max + 1, M_max + 1);
    memory_allocation_2D(v_r_check, L_max + 1, M_max + 1);
    memory_allocation_2D(v_phi_check, L_max + 1, M_max + 1);
    memory_allocation_2D(H_z_check, L_max + 1, M_max + 1);
    memory_allocation_2D(H_r_check, L_max + 1, M_max + 1);
    memory_allocation_2D(H_phi_check, L_max + 1, M_max + 1);

    // ============================================================================
    // INITIAL CONDITIONS
    // ============================================================================

    std::cout << "Setting initial conditions..." << std::endl;

#pragma omp parallel for collapse(2)
    for (int l = 0; l < L_max + 1; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            // Fluid properties
            rho[l][m] = 1.0; // Uniform initial density
            v_z[l][m] = 0.1; // Small initial axial velocity
            v_r[l][m] = 0.0; // No initial radial velocity
            v_phi[l][m] = 0; // No initial azimuthal velocity

            // Magnetic field configuration
            H_phi[l][m] =
                (1 - 0.9 * l * dz) * r_0 / r[l][m]; // Decaying azimuthal field
            H_z[l][m] = H_z0;                       // Uniform axial field
            H_r[l][m] = H_z[l][m] * r_z[l][m]; // Radial field from geometry

            // Thermodynamic properties
            e[l][m] = beta / (2.0 * (gamma - 1.0)); // Internal energy from beta
            p[l][m] = beta / 2.0;                   // Thermal pressure
            // Total pressure (thermal + magnetic)
            P[l][m] = p[l][m] + 1.0 / 2.0 *
                                    (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) +
                                     pow(H_phi[l][m], 2));

            // Initialize check arrays
            rho_check[l][m] = rho[l][m];
            v_z_check[l][m] = v_z[l][m];
            v_r_check[l][m] = v_r[l][m];
            v_phi_check[l][m] = v_phi[l][m];
            H_z_check[l][m] = H_z[l][m];
            H_r_check[l][m] = H_r[l][m];
            H_phi_check[l][m] = H_phi[l][m];
        }
    }

    // ============================================================================
    // CONVERT TO CONSERVATIVE VARIABLES
    // ============================================================================

    std::cout << "Converting to conservative variables..." << std::endl;

#pragma omp parallel for collapse(2)
    for (int l = 0; l < L_max + 1; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            u0_1[l][m] = rho[l][m] * r[l][m];             // Mass density
            u0_2[l][m] = rho[l][m] * v_z[l][m] * r[l][m]; // Axial momentum
            u0_3[l][m] = rho[l][m] * v_r[l][m] * r[l][m]; // Radial momentum
            u0_4[l][m] =
                rho[l][m] * v_phi[l][m] * r[l][m];      // Azimuthal momentum
            u0_5[l][m] = rho[l][m] * e[l][m] * r[l][m]; // Energy density
            u0_6[l][m] = H_phi[l][m];         // Azimuthal magnetic field
            u0_7[l][m] = H_z[l][m] * r[l][m]; // Axial magnetic field
            u0_8[l][m] = H_r[l][m] * r[l][m]; // Radial magnetic field
        }
    }

    // ============================================================================
    // START TIMING AND MAIN TIME LOOP
    // ============================================================================

    begin = omp_get_wtime();
    std::cout << "Starting simulation..." << std::endl;

    int total_steps = (int)(T / dt); // Total number of time steps
    int step_counter = 0;
    bool converged = false;

    // Open convergence history file
    std::ofstream conv_file("convergence_history.dat");
    conv_file << "# Step Time RelativeChange\n";

    // Main time integration loop
    while (!converged) {

        // ==========================================================================
        // FINITE DIFFERENCE SCHEME FOR INTERIOR POINTS
        // ==========================================================================

        // Lax-Friedrichs scheme: explicit finite difference method
        // u_new = 0.25*(u_neighbors) + dt*F where F contains flux and source
        // terms

#pragma omp parallel for collapse(2)
        for (int l = 1; l < L_max; l++) {
            for (int m = 1; m < M_max; m++) {

                // Mass conservation equation
                u_1[l][m] = 0.25 * (u0_1[l + 1][m] + u0_1[l - 1][m] +
                                    u0_1[l][m + 1] + u0_1[l][m - 1]) +
                            dt * (0 - // No source terms for mass conservation
                                      // Axial flux divergence
                                  (u0_1[l + 1][m] * v_z[l + 1][m] -
                                   u0_1[l - 1][m] * v_z[l - 1][m]) /
                                      (2 * dz) -
                                  // Radial flux divergence
                                  (u0_1[l][m + 1] * v_r[l][m + 1] -
                                   u0_1[l][m - 1] * v_r[l][m - 1]) /
                                      (2 * dr[l]));

                // Axial momentum conservation equation
                u_2[l][m] =
                    0.25 * (u0_2[l + 1][m] + u0_2[l - 1][m] + u0_2[l][m + 1] +
                            u0_2[l][m - 1]) +
                    dt * ( // Maxwell stress tensor - axial direction
                             ((pow(H_z[l + 1][m], 2) - P[l + 1][m]) *
                                  r[l + 1][m] -
                              (pow(H_z[l - 1][m], 2) - P[l - 1][m]) *
                                  r[l - 1][m]) /
                                 (2 * dz) +
                             // Maxwell stress tensor - radial direction
                             ((H_z[l][m + 1] * H_r[l][m + 1]) * r[l][m + 1] -
                              (H_z[l][m - 1] * H_r[l][m - 1]) * r[l][m - 1]) /
                                 (2 * dr[l]) -
                             // Convective terms
                             (u0_2[l + 1][m] * v_z[l + 1][m] -
                              u0_2[l - 1][m] * v_z[l - 1][m]) /
                                 (2 * dz) -
                             (u0_2[l][m + 1] * v_r[l][m + 1] -
                              u0_2[l][m - 1] * v_r[l][m - 1]) /
                                 (2 * dr[l]));

                // Radial momentum conservation equation
                u_3[l][m] =
                    0.25 * (u0_3[l + 1][m] + u0_3[l - 1][m] + u0_3[l][m + 1] +
                            u0_3[l][m - 1]) +
                    dt * ( // Centrifugal force + total pressure gradient -
                           // magnetic pressure
                             (rho[l][m] * pow(v_phi[l][m], 2) + P[l][m] -
                              pow(H_phi[l][m], 2)) +
                             // Maxwell stress tensor components
                             (H_z[l + 1][m] * H_r[l + 1][m] * r[l + 1][m] -
                              H_z[l - 1][m] * H_r[l - 1][m] * r[l - 1][m]) /
                                 (2 * dz) +
                             ((pow(H_r[l][m + 1], 2) - P[l][m + 1]) *
                                  r[l][m + 1] -
                              (pow(H_r[l][m - 1], 2) - P[l][m - 1]) *
                                  r[l][m - 1]) /
                                 (2 * dr[l]) -
                             // Convective terms
                             (u0_3[l + 1][m] * v_z[l + 1][m] -
                              u0_3[l - 1][m] * v_z[l - 1][m]) /
                                 (2 * dz) -
                             (u0_3[l][m + 1] * v_r[l][m + 1] -
                              u0_3[l][m - 1] * v_r[l][m - 1]) /
                                 (2 * dr[l]));

                // Azimuthal momentum conservation equation
                u_4[l][m] =
                    0.25 * (u0_4[l + 1][m] + u0_4[l - 1][m] + u0_4[l][m + 1] +
                            u0_4[l][m - 1]) +
                    dt * ( // Coriolis force + magnetic force
                             (-rho[l][m] * v_r[l][m] * v_phi[l][m] +
                              H_phi[l][m] * H_r[l][m]) +
                             // Magnetic stress terms
                             (H_phi[l + 1][m] * H_z[l + 1][m] * r[l + 1][m] -
                              H_phi[l - 1][m] * H_z[l - 1][m] * r[l - 1][m]) /
                                 (2 * dz) +
                             (H_phi[l][m + 1] * H_r[l][m + 1] * r[l][m + 1] -
                              H_phi[l][m - 1] * H_r[l][m - 1] * r[l][m - 1]) /
                                 (2 * dr[l]) -
                             // Convective terms
                             (u0_4[l + 1][m] * v_z[l + 1][m] -
                              u0_4[l - 1][m] * v_z[l - 1][m]) /
                                 (2 * dz) -
                             (u0_4[l][m + 1] * v_r[l][m + 1] -
                              u0_4[l][m - 1] * v_r[l][m - 1]) /
                                 (2 * dr[l]));

                // Energy conservation equation
                u_5[l][m] =
                    0.25 * (u0_5[l + 1][m] + u0_5[l - 1][m] + u0_5[l][m + 1] +
                            u0_5[l][m - 1]) +
                    dt * ( // P dV work term (adiabatic compression/expansion)
                             -p[l][m] * ((v_z[l + 1][m] * r[l + 1][m] -
                                          v_z[l - 1][m] * r[l - 1][m]) /
                                             (2 * dz) +
                                         (v_r[l][m + 1] * r[l][m + 1] -
                                          v_r[l][m - 1] * r[l][m - 1]) /
                                             (2 * dr[l])) -
                             // Convective energy transport
                             (u0_5[l + 1][m] * v_z[l + 1][m] -
                              u0_5[l - 1][m] * v_z[l - 1][m]) /
                                 (2 * dz) -
                             (u0_5[l][m + 1] * v_r[l][m + 1] -
                              u0_5[l][m - 1] * v_r[l][m - 1]) /
                                 (2 * dr[l]));

                // Azimuthal magnetic field evolution (induction equation)
                u_6[l][m] = 0.25 * (u0_6[l + 1][m] + u0_6[l - 1][m] +
                                    u0_6[l][m + 1] + u0_6[l][m - 1]) +
                            dt * ( // Magnetic field advection and stretching
                                     (H_z[l + 1][m] * v_phi[l + 1][m] -
                                      H_z[l - 1][m] * v_phi[l - 1][m]) /
                                         (2 * dz) +
                                     (H_r[l][m + 1] * v_phi[l][m + 1] -
                                      H_r[l][m - 1] * v_phi[l][m - 1]) /
                                         (2 * dr[l]) -
                                     // Convective transport of magnetic field
                                     (u0_6[l + 1][m] * v_z[l + 1][m] -
                                      u0_6[l - 1][m] * v_z[l - 1][m]) /
                                         (2 * dz) -
                                     (u0_6[l][m + 1] * v_r[l][m + 1] -
                                      u0_6[l][m - 1] * v_r[l][m - 1]) /
                                         (2 * dr[l]));

                // Axial magnetic field evolution
                u_7[l][m] =
                    0.25 * (u0_7[l + 1][m] + u0_7[l - 1][m] + u0_7[l][m + 1] +
                            u0_7[l][m - 1]) +
                    dt * ( // Field stretching by radial velocity
                             (H_r[l][m + 1] * v_z[l][m + 1] * r[l][m + 1] -
                              H_r[l][m - 1] * v_z[l][m - 1] * r[l][m - 1]) /
                                 (2 * dr[l]) -
                             // Convective transport
                             (u0_7[l][m + 1] * v_r[l][m + 1] -
                              u0_7[l][m - 1] * v_r[l][m - 1]) /
                                 (2 * dr[l]));

                // Radial magnetic field evolution
                u_8[l][m] =
                    0.25 * (u0_8[l + 1][m] + u0_8[l - 1][m] + u0_8[l][m + 1] +
                            u0_8[l][m - 1]) +
                    dt * ( // Field stretching by axial velocity
                             (H_z[l + 1][m] * v_r[l + 1][m] * r[l + 1][m] -
                              H_z[l - 1][m] * v_r[l - 1][m] * r[l - 1][m]) /
                                 (2 * dz) -
                             // Convective transport
                             (u0_8[l + 1][m] * v_z[l + 1][m] -
                              u0_8[l - 1][m] * v_z[l - 1][m]) /
                                 (2 * dz));
            }
        }

        // ==========================================================================
        // CONVERT CONSERVATIVE TO PRIMITIVE VARIABLES (INTERIOR)
        // ==========================================================================

#pragma omp parallel for collapse(2)
        for (int l = 1; l < L_max; l++) {
            for (int m = 1; m < M_max; m++) {
                // Extract primitive variables from conservative ones
                rho[l][m] = u_1[l][m] / r[l][m];     // Density
                v_z[l][m] = u_2[l][m] / u_1[l][m];   // Axial velocity
                v_r[l][m] = u_3[l][m] / u_1[l][m];   // Radial velocity
                v_phi[l][m] = u_4[l][m] / u_1[l][m]; // Azimuthal velocity

                H_phi[l][m] = u_6[l][m];         // Azimuthal magnetic field
                H_z[l][m] = u_7[l][m] / r[l][m]; // Axial magnetic field
                H_r[l][m] = u_8[l][m] / r[l][m]; // Radial magnetic field

                e[l][m] = u_5[l][m] / u_1[l][m]; // Specific internal energy
                p[l][m] = (gamma - 1) * rho[l][m] *
                          e[l][m]; // Thermal pressure (ideal gas)
                // Total pressure (thermal + magnetic)
                P[l][m] = p[l][m] + 1.0 / 2.0 *
                                        (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) +
                                         pow(H_phi[l][m], 2));
            }
        }

        // ==========================================================================
        // LEFT BOUNDARY CONDITIONS (z = 0) - NOW NO-PENETRATION WALL
        // ==========================================================================

#pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            // Wall boundary conditions - no penetration
            rho[0][m] = rho[1][m];     // Extrapolate density
            v_z[0][m] = 0.0;           // No axial velocity (wall)
            v_r[0][m] = 0.0;           // No radial velocity (wall)
            v_phi[0][m] = v_phi[1][m]; // Extrapolate azimuthal velocity
            e[0][m] = e[1][m];         // Extrapolate energy
            
            // Magnetic field - insulating wall conditions
            H_phi[0][m] = H_phi[1][m]; // Tangential component continuous
            H_z[0][m] = H_z[1][m];     // Tangential component continuous  
            H_r[0][m] = 0.0;           // Normal component zero at wall
        }

        // Convert boundary primitives to conservatives
#pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            u_1[0][m] = rho[0][m] * r[0][m];
            u_2[0][m] = rho[0][m] * v_z[0][m] * r[0][m];
            u_3[0][m] = rho[0][m] * v_r[0][m] * r[0][m];
            u_4[0][m] = rho[0][m] * v_phi[0][m] * r[0][m];
            u_5[0][m] = rho[0][m] * e[0][m] * r[0][m];
            u_6[0][m] = H_phi[0][m];
            u_7[0][m] = H_z[0][m] * r[0][m];
            u_8[0][m] = H_r[0][m] * r[0][m];
        }

        // ==========================================================================
        // UPPER BOUNDARY CONDITIONS (r = r_max, outer wall)
        // ==========================================================================

#pragma omp parallel for
        for (int l = 1; l < L_max; l++) {
            // Zero-gradient extrapolation from interior
            rho[l][M_max] = rho[l][M_max - 1];
            v_z[l][M_max] = v_z[l][M_max - 1];
            v_r[l][M_max] =
                v_z[l][M_max] * r_z[l][M_max]; // Tangent flow condition
            v_phi[l][M_max] = v_phi[l][M_max - 1];
            e[l][M_max] = e[l][M_max - 1];
            H_phi[l][M_max] = H_phi[l][M_max - 1];
            H_z[l][M_max] = H_z[l][M_max - 1];
            H_r[l][M_max] =
                H_z[l][M_max] * r_z[l][M_max]; // Consistent with geometry

            // Convert to conservative variables
            u_1[l][M_max] = rho[l][M_max] * r[l][M_max];
            u_2[l][M_max] = rho[l][M_max] * v_z[l][M_max] * r[l][M_max];
            u_3[l][M_max] = rho[l][M_max] * v_r[l][M_max] * r[l][M_max];
            u_4[l][M_max] = rho[l][M_max] * v_phi[l][M_max] * r[l][M_max];
            u_5[l][M_max] = rho[l][M_max] * e[l][M_max] * r[l][M_max];
            u_6[l][M_max] = H_phi[l][M_max];
            u_7[l][M_max] = H_z[l][M_max] * r[l][M_max];
            u_8[l][M_max] = H_r[l][M_max] * r[l][M_max];
        }

        // ==========================================================================
        // LOWER BOUNDARY CONDITIONS (r = r_min, inner wall/centerline)
        // ==========================================================================

        // Region 1: l <= L_end (standard wall boundary before step)
        for (int l = 1; l <= L_end; l++) {
            // Check if this is in the inlet region (step surface)
            double z_coord = l * dz;
            
            if (is_inlet_region(z_coord)) {
                // INLET BOUNDARY - Gas injection through step surface
                rho[l][0] = inlet_density;
                v_z[l][0] = 0.0;  // Primarily radial injection
                v_r[l][0] = inlet_velocity;  // Radial inflow
                v_phi[l][0] = 0.0;  // No swirl at inlet
                e[l][0] = beta / (2.0 * (gamma - 1.0));  // Fixed energy
                H_phi[l][0] = r_0 / r[l][0];  // Prescribed azimuthal field
                H_z[l][0] = H_z0;  // Constant axial field
                H_r[l][0] = 0.0;  // No radial field component
            } else {
                // WALL BOUNDARY - Zero-gradient extrapolation from interior
                rho[l][0] = rho[l][1];
                v_z[l][0] = v_z[l][1];
                v_r[l][0] = v_z[l][1] * r_z[l][1]; // Tangent flow condition
                v_phi[l][0] = v_phi[l][1];
                e[l][0] = e[l][1];
                H_phi[l][0] = H_phi[l][1];
                H_z[l][0] = H_z[l][1];
                H_r[l][0] = H_z[l][1] * r_z[l][1]; // Consistent with geometry
            }

            // Convert to conservative variables
            u_1[l][0] = rho[l][0] * r[l][0];
            u_2[l][0] = rho[l][0] * v_z[l][0] * r[l][0];
            u_3[l][0] = rho[l][0] * v_r[l][0] * r[l][0];
            u_4[l][0] = rho[l][0] * v_phi[l][0] * r[l][0];
            u_5[l][0] = rho[l][0] * e[l][0] * r[l][0];
            u_6[l][0] = H_phi[l][0];
            u_7[l][0] = H_z[l][0] * r[l][0];
            u_8[l][0] = H_r[l][0] * r[l][0];
        }

        // Region 2: l > L_end (throat region with special treatment)
#pragma omp parallel for collapse(2)
        for (int l = L_end + 1; l < L_max; l++) {
            for (int m = 0; m < 1; m++) {
                // Modified finite difference scheme for throat region
                // Mass conservation
                u_1[l][m] =
                    (0.25 *
                         (u0_1[l + 1][m] / r[l + 1][m] +
                          u0_1[l - 1][m] / r[l - 1][m] +
                          u0_1[l][m + 1] / r[l][m + 1] + u0_1[l][m] / r[l][m]) +
                     dt * (0 -
                           (u0_1[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] -
                            u0_1[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) /
                               (2 * dz) -
                           (u0_1[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] -
                            u0_1[l][m] / r[l][m] * (v_r[l][1])) /
                               (dr[l]))) *
                    r[l][m];

                // Axial momentum conservation
                u_2[l][m] =
                    (0.25 *
                         (u0_2[l + 1][m] / r[l + 1][m] +
                          u0_2[l - 1][m] / r[l - 1][m] +
                          u0_2[l][m + 1] / r[l][m + 1] + u0_2[l][m] / r[l][m]) +
                     dt * (((pow(H_z[l + 1][m], 2) - P[l + 1][m]) -
                            (pow(H_z[l - 1][m], 2) - P[l - 1][m])) /
                               (2 * dz) +
                           ((H_z[l][m + 1] * H_r[l][m + 1]) -
                            (H_z[l][m] * (H_r[l][m]))) /
                               (dr[l]) -
                           (u0_2[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] -
                            u0_2[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) /
                               (2 * dz) -
                           (u0_2[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] -
                            u0_2[l][m] / r[l][m] * (v_r[l][m])) /
                               (dr[l]))) *
                    r[l][m];

                // No radial momentum in throat
                u_3[l][m] = 0;

                // No azimuthal momentum in throat
                u_4[l][m] = 0;

                // Energy conservation
                u_5[l][m] =
                    (0.25 *
                         (u0_5[l + 1][m] / r[l + 1][m] +
                          u0_5[l - 1][m] / r[l - 1][m] +
                          u0_5[l][m + 1] / r[l][m + 1] + u0_5[l][m] / r[l][m]) +
                     dt * (-p[l][m] *
                               ((v_z[l + 1][m] - v_z[l - 1][m]) / (2 * dz) +
                                (v_r[l][m + 1] - (v_r[l][m])) / (dr[l])) -
                           (u0_5[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] -
                            u0_5[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) /
                               (2 * dz) -
                           (u0_5[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] -
                            u0_5[l][m] / r[l][m] * (v_r[l][m])) /
                               (dr[l]))) *
                    r[l][m];

                // No azimuthal magnetic field in throat
                u_6[l][m] = 0;

                // Axial magnetic field evolution
                u_7[l][m] =
                    (0.25 *
                         (u0_7[l + 1][m] / r[l + 1][m] +
                          u0_7[l - 1][m] / r[l - 1][m] +
                          u0_7[l][m + 1] / r[l][m + 1] + u0_7[l][m] / r[l][m]) +
                     dt * ((H_r[l][m + 1] * v_z[l][m + 1] -
                            (H_r[l][m]) * v_z[l][m]) /
                               (dr[l]) -
                           (u0_7[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] -
                            u0_7[l][m] / r[l][m] * (v_r[l][m])) /
                               (dr[l]))) *
                    r[l][m];

                // No radial magnetic field in throat
                u_8[l][m] = 0;
            }
        }

        // ==========================================================================
        // RIGHT BOUNDARY CONDITIONS (z = z_max, outlet)
        // ==========================================================================

#pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            // Zero-gradient extrapolation (outflow boundary)
            u_1[L_max][m] = u_1[L_max - 1][m];
            u_2[L_max][m] = u_2[L_max - 1][m];
            u_3[L_max][m] = u_3[L_max - 1][m];
            u_4[L_max][m] = u_4[L_max - 1][m];
            u_5[L_max][m] = u_5[L_max - 1][m];
            u_6[L_max][m] = u_6[L_max - 1][m];
            u_7[L_max][m] = u_7[L_max - 1][m];
            u_8[L_max][m] = u_8[L_max - 1][m];
        }

        // ==========================================================================
        // GLOBAL UPDATE: CONVERT ALL CONSERVATIVE TO PRIMITIVE VARIABLES
        // ==========================================================================

#pragma omp parallel for collapse(2)
        for (int l = 0; l < L_max + 1; l++) {
            for (int m = 0; m < M_max + 1; m++) {
                // Extract all primitive variables
                rho[l][m] = u_1[l][m] / r[l][m];
                v_z[l][m] = u_2[l][m] / u_1[l][m];
                v_r[l][m] = u_3[l][m] / u_1[l][m];
                v_phi[l][m] = u_4[l][m] / u_1[l][m];

                H_phi[l][m] = u_6[l][m];
                H_z[l][m] = u_7[l][m] / r[l][m];
                H_r[l][m] = u_8[l][m] / r[l][m];

                e[l][m] = u_5[l][m] / u_1[l][m];
                p[l][m] = (gamma - 1) * rho[l][m] * e[l][m];
                P[l][m] = p[l][m] + 1.0 / 2.0 *
                                        (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) +
                                         pow(H_phi[l][m], 2));
            }
        }

        // ==========================================================================
        // PREPARE FOR NEXT TIME STEP
        // ==========================================================================

#pragma omp parallel for collapse(2)
        for (int l = 0; l < L_max + 1; l++) {
            for (int m = 0; m < M_max + 1; m++) {
                // Copy current solution to previous time step arrays
                u0_1[l][m] = u_1[l][m];
                u0_2[l][m] = u_2[l][m];
                u0_3[l][m] = u_3[l][m];
                u0_4[l][m] = u_4[l][m];
                u0_5[l][m] = u_5[l][m];
                u0_6[l][m] = u_6[l][m];
                u0_7[l][m] = u_7[l][m];
                u0_8[l][m] = u_8[l][m];
            }
        }

        // ==========================================================================
        // CONVERGENCE CHECK
        // ==========================================================================

        if (enable_convergence_check &&
            step_counter % convergence_check_interval == 0 &&
            step_counter > 0) {

            // Compute solution change
            double solution_change = compute_solution_change(
                rho, rho_check, v_z, v_z_check, v_r, v_r_check, v_phi,
                v_phi_check, H_z, H_z_check, H_r, H_r_check, H_phi, H_phi_check,
                L_max, M_max);

            // Write to convergence history
            conv_file << step_counter << " " << t << " " << solution_change
                      << "\n";
            conv_file.flush();

            // Display convergence info
            std::cout << "\n[Convergence check at step " << step_counter
                      << ", t=" << std::fixed << std::setprecision(4) << t
                      << "] Relative change: " << std::scientific
                      << std::setprecision(3) << solution_change;

            // Check if converged
            if (solution_change < convergence_tolerance) {
                converged = true;
                std::cout << "\n\n*** CONVERGENCE REACHED ***\n";
                std::cout << "Solution converged at t = " << std::fixed
                          << std::setprecision(4) << t << " (step "
                          << step_counter << ")\n";
                std::cout << "Relative change: " << std::scientific
                          << solution_change << " < " << convergence_tolerance
                          << "\n";
                break;
            }

            // Update check arrays for next comparison
#pragma omp parallel for collapse(2)
            for (int l = 0; l < L_max + 1; l++) {
                for (int m = 0; m < M_max + 1; m++) {
                    rho_check[l][m] = rho[l][m];
                    v_z_check[l][m] = v_z[l][m];
                    v_r_check[l][m] = v_r[l][m];
                    v_phi_check[l][m] = v_phi[l][m];
                    H_z_check[l][m] = H_z[l][m];
                    H_r_check[l][m] = H_r[l][m];
                    H_phi_check[l][m] = H_phi[l][m];
                }
            }
        }

        // ==========================================================================
        // ANIMATION OUTPUT
        // ==========================================================================

        if ((int)(t * 10000) % 1000 == 0 && animate == 1) {
            animate_write((int)(t * 10000), L_max, M_max, dz, r, rho, v_z, v_r,
                          v_phi, e, H_z, H_r, H_phi);
        }

        // ==========================================================================
        // TIME ADVANCEMENT AND PROGRESS DISPLAY
        // ==========================================================================

        t += dt;
        step_counter++;

        // Display progress every 1000 steps
        if (step_counter % 1000 == 0) {
            show_progress(step_counter, total_steps, t, T);
        }
    }

    // Final progress update
    show_progress(total_steps, total_steps, T, T);
    std::cout << std::endl;

    conv_file.close();

    // ============================================================================
    // SIMULATION COMPLETED - TIMING AND OUTPUT
    // ============================================================================

    end = omp_get_wtime();
    total = end - begin;

    std::cout << "Simulation completed!" << std::endl;
    if (converged) {
        std::cout << "Reason: Convergence reached" << std::endl;
    } else {
        std::cout << "Reason: Max time reached" << std::endl;
    }
    std::cout << "Calculation time: " << std::fixed << std::setprecision(3)
              << total << " seconds" << std::endl;

    // ============================================================================
    // CALCULATING THE THRUSTER PARAMETERS
    // ============================================================================

    std::cout << "Mass flux: "
              << get_mass_flux(rho[L_max], v_z[L_max], dr[L_max], M_max)
              << std::endl;

    std::cout << "Thrust:    "
              << get_thrust(rho[L_max], v_z[L_max], p[L_max], H_r[L_max],
                            H_phi[L_max], H_z[L_max], dr[L_max], M_max)
              << std::endl;

    // ============================================================================
    // WRITE FINAL RESULTS TO FILE
    // ============================================================================

    std::cout << "Writing results to file..." << std::endl;

    std::ofstream out("28-2D_MHD_LF_rzphi_MPD_modified.plt");
    int np = (L_max + 1) * (M_max + 1); // Total number of points
    int ne = L_max * M_max;             // Total number of elements
    double hfr;                         // H_phi * r for output

    // Write Tecplot header
    out << "VARIABLES=\n";
    out << "\"z\"\n\"r\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n\"Energy\""
           "\n"
           "\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
    out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne
        << "\n ";

    // Write all grid points and solution data
    for (int m = 0; m < M_max + 1; m++) {
        for (int l = 0; l < L_max + 1; l++) {
            hfr = H_phi[l][m] * r[l][m];
            out << l * dz << " " << r[l][m] << " " << rho[l][m] << " "
                << v_z[l][m] << " " << v_r[l][m] << " "
                << std::sqrt(v_z[l][m] * v_z[l][m] + v_r[l][m] * v_r[l][m])
                << " " << v_phi[l][m] << " " << e[l][m] << " " << H_z[l][m]
                << " " << H_r[l][m] << " " << hfr << " " << H_phi[l][m] << "\n";
        }
    }

    // Write element connectivity for quadrilateral mesh
    int i1 = 0, i2 = 0, i3 = 0, i4 = 0;
    for (int m = 0; m < M_max; m++) {
        for (int l = 0; l < L_max; l++) {
            i1 = l + m * (L_max + 1) + 1;
            i2 = l + 1 + m * (L_max + 1) + 1;
            i3 = l + 1 + (m + 1) * (L_max + 1) + 1;
            i4 = l + (m + 1) * (L_max + 1) + 1;
            out << i1 << " " << i2 << " " << i3 << " " << i4 << "\n";
        }
    }

    out.close();
    std::cout << "Results written to 28-2D_MHD_LF_rzphi_MPD_modified.plt" << std::endl;

    // Write VTK file for Paraview
    write_vtk("28-2D_MHD_LF_rzphi_MPD_modified.vtk", L_max, M_max, dz, r, rho, v_z, v_r,
              v_phi, e, H_z, H_r, H_phi);
    std::cout << "Results written to 28-2D_MHD_LF_rzphi_MPD_modified.vtk" << std::endl;

    // ============================================================================
    // MEMORY CLEANUP
    // ============================================================================

    std::cout << "Cleaning up memory..." << std::endl;

    // Free conservative variable arrays
    memory_clearing_2D(u_1, L_max + 1);
    memory_clearing_2D(u_2, L_max + 1);
    memory_clearing_2D(u_3, L_max + 1);
    memory_clearing_2D(u_4, L_max + 1);
    memory_clearing_2D(u_5, L_max + 1);
    memory_clearing_2D(u_6, L_max + 1);
    memory_clearing_2D(u_7, L_max + 1);
    memory_clearing_2D(u_8, L_max + 1);

    memory_clearing_2D(u0_1, L_max + 1);
    memory_clearing_2D(u0_2, L_max + 1);
    memory_clearing_2D(u0_3, L_max + 1);
    memory_clearing_2D(u0_4, L_max + 1);
    memory_clearing_2D(u0_5, L_max + 1);
    memory_clearing_2D(u0_6, L_max + 1);
    memory_clearing_2D(u0_7, L_max + 1);
    memory_clearing_2D(u0_8, L_max + 1);

    // Free primitive variable arrays
    memory_clearing_2D(rho, L_max + 1);
    memory_clearing_2D(v_r, L_max + 1);
    memory_clearing_2D(v_phi, L_max + 1);
    memory_clearing_2D(v_z, L_max + 1);
    memory_clearing_2D(e, L_max + 1);
    memory_clearing_2D(p, L_max + 1);
    memory_clearing_2D(P, L_max + 1);
    memory_clearing_2D(H_r, L_max + 1);
    memory_clearing_2D(H_phi, L_max + 1);
    memory_clearing_2D(H_z, L_max + 1);

    // Free geometric arrays
    memory_clearing_2D(r, L_max + 1);
    memory_clearing_2D(r_z, L_max + 1);
    delete[] R;
    delete[] dr;

    // Clean up convergence check arrays
    memory_clearing_2D(rho_check, L_max + 1);
    memory_clearing_2D(v_z_check, L_max + 1);
    memory_clearing_2D(v_r_check, L_max + 1);
    memory_clearing_2D(v_phi_check, L_max + 1);
    memory_clearing_2D(H_z_check, L_max + 1);
    memory_clearing_2D(H_r_check, L_max + 1);
    memory_clearing_2D(H_phi_check, L_max + 1);

    std::cout << "Memory cleanup completed." << std::endl;

    return 0;
}
