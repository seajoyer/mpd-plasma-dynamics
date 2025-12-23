#pragma once

#include <string>

// Simulation parameters
struct SimulationParams {
    double gamma;           // Adiabatic index
    double beta;            // Plasma beta
    double H_z0;            // Initial magnetic field in z-direction
    int animate;            // Animation flag (0=off, 1=on)
    int animation_frequency; // Output every N steps
    std::string output_format;  // vtk or plt
    std::string output_dir;     // Output directory
    std::string filename_template = "default"; 

    double convergence_threshold;
    int check_frequency;
    
    double T;               // Final time
    double dt;              // Time step
    
    int L_max_global;       // Global grid size in z-direction
    int L_end;              // Boundary condition transition point
    int M_max;              // Grid size in r-direction
    
    double dz;              // Grid spacing in z
    double dy;              // Grid spacing in r (normalized)
};

// MPI domain decomposition info
struct DomainInfo {
    int rank;
    int size;
    int L_per_proc;
    int l_start;
    int l_end;
    int local_L;
    int local_L_with_ghosts;
};

// Physical field arrays (local domain with ghost cells)
struct PhysicalFields {
    double **rho;           // Density
    double **v_z;           // Velocity z-component
    double **v_r;           // Velocity r-component
    double **v_phi;         // Velocity phi-component
    double **e;             // Internal energy
    double **p;             // Pressure
    double **P;             // Total pressure (thermal + magnetic)
    double **H_z;           // Magnetic field z-component
    double **H_r;           // Magnetic field r-component
    double **H_phi;         // Magnetic field phi-component
};

// Conservative variable arrays (u_i)
struct ConservativeVars {
    double **u_1;           // rho * r
    double **u_2;           // rho * v_z * r
    double **u_3;           // rho * v_r * r
    double **u_4;           // rho * v_phi * r
    double **u_5;           // rho * e * r
    double **u_6;           // H_phi
    double **u_7;           // H_z * r
    double **u_8;           // H_r * r
};

// Grid geometry
struct GridGeometry {
    double **r;             // Radial coordinate
    double **r_z;           // Derivative of r with respect to z
    double *R;              // R = r2 - r1
    double *dr;             // Grid spacing in r-direction
};

// Previous state for convergence checking
struct PreviousState {
    double **rho_prev;
    double **v_z_prev;
    double **v_r_prev;
    double **v_phi_prev;
    double **H_z_prev;
    double **H_r_prev;
    double **H_phi_prev;
};
