#pragma once

#include <string>
#include <mpi.h>

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

// MPI domain decomposition info - Extended for 2D decomposition
struct DomainInfo {
    int rank;               // Global rank in MPI_COMM_WORLD
    int size;               // Total number of processes
    
    // Cartesian topology
    int cart_rank;          // Rank in Cartesian communicator
    int dims[2];            // Number of processes in each dimension [L, M]
    int coords[2];          // This process's coordinates in the grid [L, M]
    MPI_Comm cart_comm;     // Cartesian communicator
    
    // L-direction (axial) decomposition
    int L_per_proc;         // Base cells per process in L
    int l_start;            // Global starting L index
    int l_end;              // Global ending L index
    int local_L;            // Number of local L cells (excluding ghosts)
    int local_L_with_ghosts;// Including ghost cells
    
    // M-direction (radial) decomposition  
    int M_per_proc;         // Base cells per process in M
    int m_start;            // Global starting M index
    int m_end;              // Global ending M index
    int local_M;            // Number of local M cells (excluding ghosts)
    int local_M_with_ghosts;// Including ghost cells
    
    // Neighbor ranks in Cartesian topology (-1 if boundary)
    int neighbor_left;      // L-1 direction (z-)
    int neighbor_right;     // L+1 direction (z+)
    int neighbor_down;      // M-1 direction (r-, inner wall)
    int neighbor_up;        // M+1 direction (r+, outer wall)
    
    // Boundary flags - true if this process is at a domain boundary
    bool is_left_boundary;  // At z = 0 (inlet)
    bool is_right_boundary; // At z = z_max (outlet)
    bool is_down_boundary;  // At r = r_inner (inner wall/axis)
    bool is_up_boundary;    // At r = r_outer (outer wall)
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
    double *R;              // R = r2 - r1 (per local L index)
    double *dr;             // Grid spacing in r-direction (per local L index)
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
