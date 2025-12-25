#include "domain_decomposition.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>

void ComputeOptimalDims(int size, int L_max, int M_max, int dims[2]) {
    // Find factorization that best matches domain aspect ratio
    double domain_ratio = static_cast<double>(L_max) / static_cast<double>(M_max);
    
    int best_dims[2] = {size, 1};
    double best_ratio_diff = std::abs(static_cast<double>(size) - domain_ratio);
    
    for (int i = 1; i <= static_cast<int>(std::sqrt(size)); i++) {
        if (size % i == 0) {
            int j = size / i;
            
            // Try both orientations: (i, j) and (j, i)
            // dims[0] = L direction, dims[1] = M direction
            double ratio1 = static_cast<double>(j) / static_cast<double>(i);
            double ratio2 = static_cast<double>(i) / static_cast<double>(j);
            
            double diff1 = std::abs(ratio1 - domain_ratio);
            double diff2 = std::abs(ratio2 - domain_ratio);
            
            if (diff1 < best_ratio_diff) {
                best_ratio_diff = diff1;
                best_dims[0] = j;  // More processes in L direction
                best_dims[1] = i;
            }
            if (diff2 < best_ratio_diff) {
                best_ratio_diff = diff2;
                best_dims[0] = i;
                best_dims[1] = j;
            }
        }
    }
    
    dims[0] = best_dims[0];
    dims[1] = best_dims[1];
}

void SetupCartesianTopology(DomainInfo& domain, int L_max, int M_max) {
    // Get basic MPI info first
    MPI_Comm_rank(MPI_COMM_WORLD, &domain.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &domain.size);
    
    // Compute optimal grid dimensions
    ComputeOptimalDims(domain.size, L_max, M_max, domain.dims);
    
    // Create 2D Cartesian topology
    // periods = {0, 0} - no periodic boundaries
    int periods[2] = {0, 0};
    int reorder = 1;  // Allow MPI to reorder ranks for efficiency
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, domain.dims, periods, reorder, &domain.cart_comm);
    
    // Get this process's coordinates and rank in the Cartesian grid
    MPI_Comm_rank(domain.cart_comm, &domain.cart_rank);
    MPI_Cart_coords(domain.cart_comm, domain.cart_rank, 2, domain.coords);
    
    // Get neighbor ranks using MPI_Cart_shift
    // L direction (dimension 0): left = -1, right = +1
    MPI_Cart_shift(domain.cart_comm, 0, 1, &domain.neighbor_left, &domain.neighbor_right);
    // M direction (dimension 1): down = -1, up = +1
    MPI_Cart_shift(domain.cart_comm, 1, 1, &domain.neighbor_down, &domain.neighbor_up);
    
    // Set boundary flags based on neighbor existence
    // MPI_Cart_shift returns MPI_PROC_NULL (-2 typically) for non-existent neighbors
    domain.is_left_boundary = (domain.neighbor_left == MPI_PROC_NULL);
    domain.is_right_boundary = (domain.neighbor_right == MPI_PROC_NULL);
    domain.is_down_boundary = (domain.neighbor_down == MPI_PROC_NULL);
    domain.is_up_boundary = (domain.neighbor_up == MPI_PROC_NULL);
}

void ComputeLocalDomainExtents(DomainInfo& domain, int L_max_global, int M_max) {
    // Compute L-direction decomposition
    domain.L_per_proc = L_max_global / domain.dims[0];
    int L_remainder = L_max_global % domain.dims[0];
    
    // Distribute remainder to first L_remainder processes in L direction
    if (domain.coords[0] < L_remainder) {
        domain.local_L = domain.L_per_proc + 1;
        domain.l_start = domain.coords[0] * (domain.L_per_proc + 1);
    } else {
        domain.local_L = domain.L_per_proc;
        domain.l_start = L_remainder * (domain.L_per_proc + 1) + 
                         (domain.coords[0] - L_remainder) * domain.L_per_proc;
    }
    domain.l_end = domain.l_start + domain.local_L - 1;
    domain.local_L_with_ghosts = domain.local_L + 2;  // One ghost on each side
    
    // Compute M-direction decomposition
    int M_total = M_max + 1;  // M goes from 0 to M_max inclusive
    domain.M_per_proc = M_total / domain.dims[1];
    int M_remainder = M_total % domain.dims[1];
    
    // Distribute remainder to first M_remainder processes in M direction
    if (domain.coords[1] < M_remainder) {
        domain.local_M = domain.M_per_proc + 1;
        domain.m_start = domain.coords[1] * (domain.M_per_proc + 1);
    } else {
        domain.local_M = domain.M_per_proc;
        domain.m_start = M_remainder * (domain.M_per_proc + 1) + 
                         (domain.coords[1] - M_remainder) * domain.M_per_proc;
    }
    domain.m_end = domain.m_start + domain.local_M - 1;
    domain.local_M_with_ghosts = domain.local_M + 2;  // One ghost on each side
}

void Setup2DDecomposition(DomainInfo& domain, const SimulationParams& params) {
    // Set up Cartesian topology
    SetupCartesianTopology(domain, params.L_max_global, params.M_max + 1);
    
    // Compute local domain extents
    ComputeLocalDomainExtents(domain, params.L_max_global, params.M_max);
}

void PrintDecompositionInfo(const DomainInfo& domain, const SimulationParams& params) {
    // Print from each rank in order
    for (int r = 0; r < domain.size; r++) {
        if (domain.rank == r) {
            printf("Rank %d (cart_rank %d): coords=(%d,%d)\n",
                   domain.rank, domain.cart_rank, domain.coords[0], domain.coords[1]);
            printf("  L: [%d,%d] (%d cells + 2 ghosts)\n",
                   domain.l_start, domain.l_end, domain.local_L);
            printf("  M: [%d,%d] (%d cells + 2 ghosts)\n",
                   domain.m_start, domain.m_end, domain.local_M);
            printf("  Neighbors: left=%d, right=%d, down=%d, up=%d\n",
                   domain.neighbor_left, domain.neighbor_right,
                   domain.neighbor_down, domain.neighbor_up);
            printf("  Boundaries: left=%d, right=%d, down=%d, up=%d\n",
                   domain.is_left_boundary, domain.is_right_boundary,
                   domain.is_down_boundary, domain.is_up_boundary);
            fflush(stdout);
        }
        MPI_Barrier(domain.cart_comm);
    }
    
    if (domain.rank == 0) {
        printf("\n2D Decomposition: %d x %d processes for %d x %d domain\n",
               domain.dims[0], domain.dims[1], params.L_max_global, params.M_max + 1);
    }
}
