#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <iomanip>

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
    if (z < 0.3) {
        return 0.2; // Constant radius in the initial section
    } else if (z >= 0.3 && z < 0.4) {
        return 0.2 - 10 * pow((z - 0.3), 2); // Quadratic constriction
    } else if (z >= 0.4 && z < 0.478) {
        return 10 * pow((z - 0.5), 2); // Quadratic expansion
    } else {
        return 0.005; // Narrow throat region
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
    if (z < 0.3) {
        return 0; // No slope in constant region
    } else if (z >= 0.3 && z < 0.4) {
        return -10 * 2 * (z - 0.3); // Negative slope (constriction)
    } else if (z >= 0.4 && z < 0.478) {
        return 10 * 2 * (z - 0.5); // Positive slope (expansion)
    } else {
        return 0; // No slope in throat region
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
    char filename[10];
    sprintf(filename, "animate_m_29_800x400/%d.plt", n);

    std::ofstream out(filename);
    int np = (L_max + 1) * (M_max + 1); // Total number of points
    int ne = L_max * M_max; // Total number of elements
    double hfr; // H_phi * r for output

    // Write Tecplot header
    out << "VARIABLES=\n";
    out << "\"z\"\n\"r\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n\"Energy\"\n"
            "\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
    out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne
            << "\n ";

    // Write point data
    for (int m = 0; m < M_max + 1; m++) {
        for (int l = 0; l < L_max + 1; l++) {
            hfr = H_phi[l][m] * r[l][m];
            out << l * dz << " " << r[l][m] << " " << rho[l][m] << " " << v_z[l][m]
                    << " " << v_r[l][m] << " "
                    << std::sqrt(v_z[l][m] * v_z[l][m] + v_r[l][m] * v_r[l][m]) << " "
                    << v_phi[l][m] << " " << e[l][m] << " " << H_z[l][m] << " "
                    << H_r[l][m] << " " << hfr << " " << H_phi[l][m] << "\n";
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
void write_vtk(const char *filename, int L_max, int M_max, double dz, double **r,
               double **rho, double **v_z, double **v_r, double **v_phi,
               double **e, double **H_z, double **H_r, double **H_phi) {
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
    for (int j = 0; j < ny; ++j) {
        // m (r)
        for (int i = 0; i < nx; ++i) {
            // l (z)
            double x = r[i][j]; // r[l][m]
            double y = i * dz; // z = l * dz
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
            double vl = std::sqrt(v_z[i][j] * v_z[i][j] + v_r[i][j] * v_r[i][j]);
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
 * Writes simulation data for v_z to see acceleration
 * @param filename Output filename
 * @param L_max Maximum number of axial grid points
 * @param M_max Maximum number of radial grid points
 * @param dz Axial grid spacing
 * @param r Radial coordinate array
 * @param v_z Axial velocity array
 */
void write_csv(const char *filename, const int L_max, const int M_max, const double dz, double **r, double **v_z) {
    std::ofstream out(filename);
    out << "z,r,v_z" << std::endl;
    for (int m = 0; m < M_max + 1; m++) {
        for (int l = 0; l < L_max + 1; l++) {
            out << l * dz << "," << r[l][m] << "," << v_z[l][m] << "\n";
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
void show_progress(int current, int total, double current_time, double total_time) {
    const int bar_width = 50;
    double progress = (double) current / total;

    std::cout << "\r[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1)
            << progress * 100.0 << "% | Time: "
            << std::setprecision(4) << current_time << "/" << total_time;
    std::cout.flush();
}

int main(int argc, char *argv[]) {
    // ============================================================================
    // PHYSICAL PARAMETERS AND SIMULATION SETUP
    // ============================================================================

    double gamma = 1.67; // Adiabatic index (ratio of specific heats)
    double beta = 0.05; // Plasma beta parameter (thermal to magnetic pressure ratio)
    double H_z0 = 0.25; // Initial axial magnetic field strength

    int animate = 0; // Animation flag (0 = off, 1 = on)

    // ============================================================================
    // COMPUTATIONAL DOMAIN AND TIME DISCRETIZATION
    // ============================================================================

    double T = 10.0; // Total simulation time
    double t = 0.0; // Current time
    double dt = 0.00005; // Time step size

    int L_max = 400; // Number of axial grid points
    int L_end = 160; // Transition point for boundary conditions
    int M_max = 200; // Number of radial grid points

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
    //                            rho*r*v_phi*R, rho*r*e*R, H_phi*R, H_z*r*R, H_y*r}
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

    double **rho; // Mass density
    double **v_r; // Radial velocity
    double **v_phi; // Azimuthal velocity
    double **v_z; // Axial velocity
    double **e; // Specific internal energy
    double **p; // Thermal pressure
    double **P; // Total pressure (thermal + magnetic)
    double **H_r; // Radial magnetic field
    double **H_phi; // Azimuthal magnetic field
    double **H_z; // Axial magnetic field

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

    double **r; // Radial coordinate at each grid point
    double **r_z; // Derivative dr/dz at each grid point
    memory_allocation_2D(r, L_max + 1, M_max + 1);
    memory_allocation_2D(r_z, L_max + 1, M_max + 1);

    double *R = new double[L_max + 1]; // Local domain width R(z) = r2(z) - r1(z)
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
        dr[l] = R[l] / M_max; // Local radial spacing

        for (int m = 0; m < M_max + 1; m++) {
            // Linear interpolation between inner and outer boundaries
            r[l][m] = (1 - m * dy) * r1(l * dz) + m * dy * r2(l * dz);
            // Derivative of radius with respect to z
            r_z[l][m] = (1 - m * dy) * der_r1(l * dz) + m * dy * der_r2(l * dz);
        }
    }

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
            v_r[l][m] = 0.1; // Small initial radial velocity
            v_phi[l][m] = 0; // No initial azimuthal velocity

            // Magnetic field configuration
            H_phi[l][m] = (1 - 0.9 * l * dz) * r_0 / r[l][m]; // Decaying azimuthal field
            H_z[l][m] = H_z0; // Uniform axial field
            H_r[l][m] = H_z[l][m] * r_z[l][m]; // Radial field from geometry

            // Thermodynamic properties
            e[l][m] = beta / (2.0 * (gamma - 1.0)); // Internal energy from beta
            p[l][m] = beta / 2.0; // Thermal pressure
            // Total pressure (thermal + magnetic)
            P[l][m] = p[l][m] + 1.0 / 2.0 * (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) +
                                             pow(H_phi[l][m], 2));
        }
    }

    // ============================================================================
    // CONVERT TO CONSERVATIVE VARIABLES
    // ============================================================================

    std::cout << "Converting to conservative variables..." << std::endl;

#pragma omp parallel for collapse(2)
    for (int l = 0; l < L_max + 1; l++) {
        for (int m = 0; m < M_max + 1; m++) {
            u0_1[l][m] = rho[l][m] * r[l][m]; // Mass density
            u0_2[l][m] = rho[l][m] * v_z[l][m] * r[l][m]; // Axial momentum
            u0_3[l][m] = rho[l][m] * v_r[l][m] * r[l][m]; // Radial momentum
            u0_4[l][m] = rho[l][m] * v_phi[l][m] * r[l][m]; // Azimuthal momentum
            u0_5[l][m] = rho[l][m] * e[l][m] * r[l][m]; // Energy density
            u0_6[l][m] = H_phi[l][m]; // Azimuthal magnetic field
            u0_7[l][m] = H_z[l][m] * r[l][m]; // Axial magnetic field
            u0_8[l][m] = H_r[l][m] * r[l][m]; // Radial magnetic field
        }
    }

    // ============================================================================
    // START TIMING AND MAIN TIME LOOP
    // ============================================================================

    begin = omp_get_wtime();
    std::cout << "Starting simulation..." << std::endl;

    int total_steps = (int) (T / dt); // Total number of time steps
    int step_counter = 0;

    // Main time integration loop
    while (t < T) {
        // ==========================================================================
        // FINITE DIFFERENCE SCHEME FOR INTERIOR POINTS
        // ==========================================================================

        // Lax-Friedrichs scheme: explicit finite difference method
        // u_new = 0.25*(u_neighbors) + dt*F where F contains flux and source terms

#pragma omp parallel for collapse(2)
        for (int l = 1; l < L_max; l++) {
            for (int m = 1; m < M_max; m++) {
                // Mass conservation equation
                u_1[l][m] = 0.25 * (u0_1[l + 1][m] + u0_1[l - 1][m] + u0_1[l][m + 1] +
                                    u0_1[l][m - 1]) +
                            dt * (0 - // No source terms for mass conservation
                                  // Axial flux divergence
                                  (u0_1[l + 1][m] * v_z[l + 1][m] -
                                   u0_1[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                                  // Radial flux divergence
                                  (u0_1[l][m + 1] * v_r[l][m + 1] -
                                   u0_1[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

                // Axial momentum conservation equation
                u_2[l][m] =
                        0.25 * (u0_2[l + 1][m] + u0_2[l - 1][m] + u0_2[l][m + 1] +
                                u0_2[l][m - 1]) +
                        dt * ( // Maxwell stress tensor - axial direction
                            ((pow(H_z[l + 1][m], 2) - P[l + 1][m]) * r[l + 1][m] -
                             (pow(H_z[l - 1][m], 2) - P[l - 1][m]) * r[l - 1][m]) / (2 * dz) +
                            // Maxwell stress tensor - radial direction
                            ((H_z[l][m + 1] * H_r[l][m + 1]) * r[l][m + 1] -
                             (H_z[l][m - 1] * H_r[l][m - 1]) * r[l][m - 1]) / (2 * dr[l]) -
                            // Convective terms
                            (u0_2[l + 1][m] * v_z[l + 1][m] -
                             u0_2[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                            (u0_2[l][m + 1] * v_r[l][m + 1] -
                             u0_2[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

                // Radial momentum conservation equation
                u_3[l][m] =
                        0.25 * (u0_3[l + 1][m] + u0_3[l - 1][m] + u0_3[l][m + 1] +
                                u0_3[l][m - 1]) +
                        dt * ( // Centrifugal force + total pressure gradient - magnetic pressure
                            (rho[l][m] * pow(v_phi[l][m], 2) + P[l][m] - pow(H_phi[l][m], 2)) +
                            // Maxwell stress tensor components
                            (H_z[l + 1][m] * H_r[l + 1][m] * r[l + 1][m] -
                             H_z[l - 1][m] * H_r[l - 1][m] * r[l - 1][m]) / (2 * dz) +
                            ((pow(H_r[l][m + 1], 2) - P[l][m + 1]) * r[l][m + 1] -
                             (pow(H_r[l][m - 1], 2) - P[l][m - 1]) * r[l][m - 1]) / (2 * dr[l]) -
                            // Convective terms
                            (u0_3[l + 1][m] * v_z[l + 1][m] -
                             u0_3[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                            (u0_3[l][m + 1] * v_r[l][m + 1] -
                             u0_3[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

                // Azimuthal momentum conservation equation
                u_4[l][m] = 0.25 * (u0_4[l + 1][m] + u0_4[l - 1][m] + u0_4[l][m + 1] +
                                    u0_4[l][m - 1]) +
                            dt * ( // Coriolis force + magnetic force
                                (-rho[l][m] * v_r[l][m] * v_phi[l][m] +
                                 H_phi[l][m] * H_r[l][m]) +
                                // Magnetic stress terms
                                (H_phi[l + 1][m] * H_z[l + 1][m] * r[l + 1][m] -
                                 H_phi[l - 1][m] * H_z[l - 1][m] * r[l - 1][m]) / (2 * dz) +
                                (H_phi[l][m + 1] * H_r[l][m + 1] * r[l][m + 1] -
                                 H_phi[l][m - 1] * H_r[l][m - 1] * r[l][m - 1]) / (2 * dr[l]) -
                                // Convective terms
                                (u0_4[l + 1][m] * v_z[l + 1][m] -
                                 u0_4[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                                (u0_4[l][m + 1] * v_r[l][m + 1] -
                                 u0_4[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

                // Energy conservation equation
                u_5[l][m] = 0.25 * (u0_5[l + 1][m] + u0_5[l - 1][m] + u0_5[l][m + 1] +
                                    u0_5[l][m - 1]) +
                            dt * ( // P dV work term (adiabatic compression/expansion)
                                -p[l][m] * ((v_z[l + 1][m] * r[l + 1][m] -
                                             v_z[l - 1][m] * r[l - 1][m]) / (2 * dz) +
                                            (v_r[l][m + 1] * r[l][m + 1] -
                                             v_r[l][m - 1] * r[l][m - 1]) / (2 * dr[l])) -
                                // Convective energy transport
                                (u0_5[l + 1][m] * v_z[l + 1][m] -
                                 u0_5[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                                (u0_5[l][m + 1] * v_r[l][m + 1] -
                                 u0_5[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

                // Azimuthal magnetic field evolution (induction equation)
                u_6[l][m] = 0.25 * (u0_6[l + 1][m] + u0_6[l - 1][m] + u0_6[l][m + 1] +
                                    u0_6[l][m - 1]) +
                            dt * ( // Magnetic field advection and stretching
                                (H_z[l + 1][m] * v_phi[l + 1][m] -
                                 H_z[l - 1][m] * v_phi[l - 1][m]) / (2 * dz) +
                                (H_r[l][m + 1] * v_phi[l][m + 1] -
                                 H_r[l][m - 1] * v_phi[l][m - 1]) / (2 * dr[l]) -
                                // Convective transport of magnetic field
                                (u0_6[l + 1][m] * v_z[l + 1][m] -
                                 u0_6[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                                (u0_6[l][m + 1] * v_r[l][m + 1] -
                                 u0_6[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

                // Axial magnetic field evolution
                u_7[l][m] = 0.25 * (u0_7[l + 1][m] + u0_7[l - 1][m] + u0_7[l][m + 1] +
                                    u0_7[l][m - 1]) +
                            dt * ( // Field stretching by radial velocity
                                (H_r[l][m + 1] * v_z[l][m + 1] * r[l][m + 1] -
                                 H_r[l][m - 1] * v_z[l][m - 1] * r[l][m - 1]) / (2 * dr[l]) -
                                // Convective transport
                                (u0_7[l][m + 1] * v_r[l][m + 1] -
                                 u0_7[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

                // Radial magnetic field evolution
                u_8[l][m] = 0.25 * (u0_8[l + 1][m] + u0_8[l - 1][m] + u0_8[l][m + 1] +
                                    u0_8[l][m - 1]) +
                            dt * ( // Field stretching by axial velocity
                                (H_z[l + 1][m] * v_r[l + 1][m] * r[l + 1][m] -
                                 H_z[l - 1][m] * v_r[l - 1][m] * r[l - 1][m]) / (2 * dz) -
                                // Convective transport
                                (u0_8[l + 1][m] * v_z[l + 1][m] -
                                 u0_8[l - 1][m] * v_z[l - 1][m]) / (2 * dz));
            }
        }

        // ==========================================================================
        // CONVERT CONSERVATIVE TO PRIMITIVE VARIABLES (INTERIOR)
        // ==========================================================================

#pragma omp parallel for collapse(2)
        for (int l = 1; l < L_max; l++) {
            for (int m = 1; m < M_max; m++) {
                // Extract primitive variables from conservative ones
                rho[l][m] = u_1[l][m] / r[l][m]; // Density
                v_z[l][m] = u_2[l][m] / u_1[l][m]; // Axial velocity
                v_r[l][m] = u_3[l][m] / u_1[l][m]; // Radial velocity
                v_phi[l][m] = u_4[l][m] / u_1[l][m]; // Azimuthal velocity

                H_phi[l][m] = u_6[l][m]; // Azimuthal magnetic field
                H_z[l][m] = u_7[l][m] / r[l][m]; // Axial magnetic field
                H_r[l][m] = u_8[l][m] / r[l][m]; // Radial magnetic field

                e[l][m] = u_5[l][m] / u_1[l][m]; // Specific internal energy
                p[l][m] = (gamma - 1) * rho[l][m] * e[l][m]; // Thermal pressure (ideal gas)
                // Total pressure (thermal + magnetic)
                P[l][m] = p[l][m] + 1.0 / 2.0 * (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) +
                                                 pow(H_phi[l][m], 2));
            }
        }

        // ==========================================================================
        // LEFT BOUNDARY CONDITIONS (z = 0, inlet)
        // ==========================================================================

#pragma omp parallel for
        for (int m = 0; m < M_max + 1; m++) {
            // Fixed inlet conditions
            rho[0][m] = 1.0; // Constant density
            v_phi[0][m] = 0; // No swirl at inlet
            v_z[0][m] = u_2[1][m] / (rho[0][m] * r[0][m]); // Extrapolated axial velocity
            v_r[0][m] = 0; // No radial velocity at inlet
            H_phi[0][m] = r_0 / r[0][m]; // Prescribed azimuthal field
            H_z[0][m] = H_z0; // Constant axial field
            H_r[0][m] = 0; // No radial field at inlet
            // Isentropic relation for energy
            e[0][m] = beta / (2.0 * (gamma - 1.0)) * pow(rho[0][m], gamma - 1.0);
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
            v_r[l][M_max] = v_z[l][M_max] * r_z[l][M_max]; // Tangent flow condition
            v_phi[l][M_max] = v_phi[l][M_max - 1];
            e[l][M_max] = e[l][M_max - 1];
            H_phi[l][M_max] = H_phi[l][M_max - 1];
            H_z[l][M_max] = H_z[l][M_max - 1];
            H_r[l][M_max] = H_z[l][M_max] * r_z[l][M_max]; // Consistent with geometry

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

        // Region 1: l <= L_end (standard wall boundary)
        for (int l = 1; l <= L_end; l++) {
            // Zero-gradient extrapolation from interior
            rho[l][0] = rho[l][1];
            v_z[l][0] = v_z[l][1];
            v_r[l][0] = v_z[l][1] * r_z[l][1]; // Tangent flow condition
            v_phi[l][0] = v_phi[l][1];
            e[l][0] = e[l][1];
            H_phi[l][0] = H_phi[l][1];
            H_z[l][0] = H_z[l][1];
            H_r[l][0] = H_z[l][1] * r_z[l][1]; // Consistent with geometry

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
                         (u0_1[l + 1][m] / r[l + 1][m] + u0_1[l - 1][m] / r[l - 1][m] +
                          u0_1[l][m + 1] / r[l][m + 1] + u0_1[l][m] / r[l][m]) +
                         dt * (0 -
                               (u0_1[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] -
                                u0_1[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                               (u0_1[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] -
                                u0_1[l][m] / r[l][m] * (v_r[l][1])) / (dr[l]))) *
                        r[l][m];

                // Axial momentum conservation
                u_2[l][m] =
                        (0.25 *
                         (u0_2[l + 1][m] / r[l + 1][m] + u0_2[l - 1][m] / r[l - 1][m] +
                          u0_2[l][m + 1] / r[l][m + 1] + u0_2[l][m] / r[l][m]) +
                         dt * (((pow(H_z[l + 1][m], 2) - P[l + 1][m]) -
                                (pow(H_z[l - 1][m], 2) - P[l - 1][m])) / (2 * dz) +
                               ((H_z[l][m + 1] * H_r[l][m + 1]) -
                                (H_z[l][m] * (H_r[l][m]))) / (dr[l]) -
                               (u0_2[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] -
                                u0_2[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                               (u0_2[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] -
                                u0_2[l][m] / r[l][m] * (v_r[l][m])) / (dr[l]))) *
                        r[l][m];

                // No radial momentum in throat
                u_3[l][m] = 0;

                // No azimuthal momentum in throat
                u_4[l][m] = 0;

                // Energy conservation
                u_5[l][m] =
                        (0.25 *
                         (u0_5[l + 1][m] / r[l + 1][m] + u0_5[l - 1][m] / r[l - 1][m] +
                          u0_5[l][m + 1] / r[l][m + 1] + u0_5[l][m] / r[l][m]) +
                         dt * (-p[l][m] * ((v_z[l + 1][m] - v_z[l - 1][m]) / (2 * dz) +
                                           (v_r[l][m + 1] - (v_r[l][m])) / (dr[l])) -
                               (u0_5[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] -
                                u0_5[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) / (2 * dz) -
                               (u0_5[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] -
                                u0_5[l][m] / r[l][m] * (v_r[l][m])) / (dr[l]))) *
                        r[l][m];

                // No azimuthal magnetic field in throat
                u_6[l][m] = 0;

                // Axial magnetic field evolution
                u_7[l][m] =
                        (0.25 *
                         (u0_7[l + 1][m] / r[l + 1][m] + u0_7[l - 1][m] / r[l - 1][m] +
                          u0_7[l][m + 1] / r[l][m + 1] + u0_7[l][m] / r[l][m]) +
                         dt * ((H_r[l][m + 1] * v_z[l][m + 1] - (H_r[l][m]) * v_z[l][m]) /
                               (dr[l]) -
                               (u0_7[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] -
                                u0_7[l][m] / r[l][m] * (v_r[l][m])) / (dr[l]))) *
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
                P[l][m] = p[l][m] + 1.0 / 2.0 * (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) +
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
        // ANIMATION OUTPUT (if enabled)
        // ==========================================================================

        if ((int) (t * 10000) % 1000 == 0 && animate == 1) {
            animate_write((int) (t * 10000), L_max, M_max, dz, r, rho, v_z, v_r, v_phi,
                          e, H_z, H_r, H_phi);
        }

        // ==========================================================================
        // TIME ADVANCEMENT AND PROGRESS DISPLAY
        // ==========================================================================

        t += dt; // Advance simulation time
        step_counter++; // Increment step counter

        // Display progress every 1000 steps
        if (step_counter % 1000 == 0) {
            show_progress(step_counter, total_steps, t, T);
        }
    }

    // Final progress update
    show_progress(total_steps, total_steps, T, T);
    std::cout << std::endl;

    // ============================================================================
    // SIMULATION COMPLETED - TIMING AND OUTPUT
    // ============================================================================

    end = omp_get_wtime();
    total = end - begin;
    std::cout << "Simulation completed!" << std::endl;
    std::cout << "Calculation time: " << std::fixed << std::setprecision(3)
            << total << " seconds" << std::endl;

    // ============================================================================
    // WRITE FINAL RESULTS TO FILE
    // ============================================================================

    std::cout << "Writing results to file..." << std::endl;

    std::ofstream out("28-2D_MHD_LF_rzphi_MPD.plt");
    int np = (L_max + 1) * (M_max + 1); // Total number of points
    int ne = L_max * M_max; // Total number of elements
    double hfr; // H_phi * r for output

    // Write Tecplot header
    out << "VARIABLES=\n";
    out << "\"z\"\n\"r\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n\"Energy\"\n"
            "\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
    out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne
            << "\n ";

    // Write all grid points and solution data
    for (int m = 0; m < M_max + 1; m++) {
        for (int l = 0; l < L_max + 1; l++) {
            hfr = H_phi[l][m] * r[l][m];
            out << l * dz << " " << r[l][m] << " " << rho[l][m] << " " << v_z[l][m]
                    << " " << v_r[l][m] << " "
                    << std::sqrt(v_z[l][m] * v_z[l][m] + v_r[l][m] * v_r[l][m]) << " "
                    << v_phi[l][m] << " " << e[l][m] << " " << H_z[l][m] << " "
                    << H_r[l][m] << " " << hfr << " " << H_phi[l][m] << "\n";
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
    std::cout << "Results written to 28-2D_MHD_LF_rzphi_MPD.plt" << std::endl;

    // Write VTK file for Paraview
    write_vtk("28-2D_MHD_LF_rzphi_MPD.vtk", L_max, M_max, dz, r, rho, v_z, v_r, v_phi,
              e, H_z, H_r, H_phi);
    std::cout << "Results written to 28-2D_MHD_LF_rzphi_MPD.vtk" << std::endl;

    write_csv("v_z.csv", L_max, M_max, dz, r, v_z);

    std::cout << "Results written to v_z.csv" << std::endl;
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

    std::cout << "Memory cleanup completed." << std::endl;

    return 0;
}
