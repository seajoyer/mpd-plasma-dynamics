#include "types.hpp"
#include <mpi.h>
#include <cstdio>
#include "domain_decomposition.hpp"

void SetupCartesianTopology(DomainInfo& domain, int total_procs) {
    // Initialize dimensions to 0 (MPI will determine optimal split)
    domain.dims[0] = 0;  // P_L (L-direction)
    domain.dims[1] = 0;  // P_M (M-direction)
    
    // Let MPI determine optimal 2D decomposition
    MPI_Dims_create(total_procs, 2, domain.dims);
    
    // Optional: Manually optimize for L:M aspect ratio (800:400 = 2:1)
    // Prefer more processes in L-direction
    if (domain.dims[0] < domain.dims[1]) {
        int temp = domain.dims[0];
        domain.dims[0] = domain.dims[1];
        domain.dims[1] = temp;
    }
    
    // Create Cartesian communicator (non-periodic boundaries)
    int periods[2] = {0, 0};
    int reorder = 1;  // Allow MPI to reorder ranks for better performance
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, domain.dims, periods, reorder, &domain.cart_comm);
    
    // Get this process's rank and coordinates in the Cartesian grid
    MPI_Comm_rank(domain.cart_comm, &domain.rank);
    MPI_Cart_coords(domain.cart_comm, domain.rank, 2, domain.coords);
    
    // Get neighbor ranks
    MPI_Cart_shift(domain.cart_comm, 0, 1, &domain.rank_left, &domain.rank_right);
    MPI_Cart_shift(domain.cart_comm, 1, 1, &domain.rank_down, &domain.rank_up);
    
    // Store total number of processes
    MPI_Comm_size(domain.cart_comm, &domain.size);
    
    if (domain.rank == 0) {
        printf("=== 2D Domain Decomposition Setup ===\n");
        printf("Total processes: %d\n", total_procs);
        printf("Process grid: %d x %d (L x M)\n", domain.dims[0], domain.dims[1]);
        printf("====================================\n\n");
    }
}

void SetupDomainDecomposition(DomainInfo& domain, SimulationParams& params) {
    // L-dimension decomposition
    domain.L_per_proc = params.L_max_global / domain.dims[0];
    domain.l_start = domain.coords[0] * domain.L_per_proc;
    domain.l_end = (domain.coords[0] + 1) * domain.L_per_proc - 1;
    
    // Handle remainder cells for last process in L-direction
    if (domain.coords[0] == domain.dims[0] - 1) {
        domain.l_end = params.L_max_global - 1;
    }
    
    domain.local_L = domain.l_end - domain.l_start + 1;
    domain.local_L_with_ghosts = domain.local_L + 2;
    
    // M-dimension decomposition
    domain.M_per_proc = params.M_max / domain.dims[1];
    domain.m_start = domain.coords[1] * domain.M_per_proc;
    domain.m_end = (domain.coords[1] + 1) * domain.M_per_proc - 1;
    
    // Handle remainder cells for last process in M-direction
    if (domain.coords[1] == domain.dims[1] - 1) {
        domain.m_end = params.M_max;
    }
    
    domain.local_M = domain.m_end - domain.m_start + 1;
    domain.local_M_with_ghosts = domain.local_M + 2;
    
    // Debug output
    if (domain.rank == 0) {
        printf("Rank %d: coords=(%d,%d), L=[%d,%d] (%d cells), M=[%d,%d] (%d cells)\n",
               domain.rank, domain.coords[0], domain.coords[1],
               domain.l_start, domain.l_end, domain.local_L,
               domain.m_start, domain.m_end, domain.local_M);
    }
}

void PrintDomainInfo(const DomainInfo& domain) {
    // Each rank prints its domain info (synchronized)
    for (int r = 0; r < domain.size; r++) {
        if (domain.rank == r) {
            printf("Rank %2d: coords=(%d,%d) | L=[%3d,%3d] (%3d) | M=[%3d,%3d] (%3d) | "
                   "Neighbors: L=%2d R=%2d D=%2d U=%2d\n",
                   domain.rank, domain.coords[0], domain.coords[1],
                   domain.l_start, domain.l_end, domain.local_L,
                   domain.m_start, domain.m_end, domain.local_M,
                   domain.rank_left, domain.rank_right, domain.rank_down, domain.rank_up);
            fflush(stdout);
        }
        MPI_Barrier(domain.cart_comm);
    }
}
