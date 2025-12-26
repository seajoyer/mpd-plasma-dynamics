#pragma once

#include "types.hpp"

// Helper function to compute balanced domain decomposition
// Distributes remainder cells evenly across first N processes
inline void ComputeBalancedDecomposition(int L_max_global, int size, int rank,
                                         int& l_start, int& l_end, int& local_L) {
    int L_per_proc = L_max_global / size;
    int remainder = L_max_global % size;
    
    // Processes with rank < remainder get one extra cell
    if (rank < remainder) {
        local_L = L_per_proc + 1;
        l_start = rank * local_L;
    } else {
        local_L = L_per_proc;
        l_start = remainder * (L_per_proc + 1) + (rank - remainder) * L_per_proc;
    }
    
    l_end = l_start + local_L - 1;
}

// Helper function to compute decomposition info for any rank
inline void GetDecompositionForRank(int L_max_global, int size, int target_rank,
                                    int& l_start, int& l_end, int& local_L) {
    ComputeBalancedDecomposition(L_max_global, size, target_rank, l_start, l_end, local_L);
}
