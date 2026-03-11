#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

void SimConfig::init() {
    dz = 1.0 / L_max_global;
    dy = 1.0 / M_max;
}

void SimConfig::parse_args(int argc, char* argv[], int rank) {
    // argv[1] is the OpenMP thread count (legacy positional arg)
    for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--converge") {
            convergence_threshold = std::atof(argv[i + 1]);
            if (rank == 0)
                printf("Convergence checking enabled: threshold = %e\n",
                       convergence_threshold);
            break;
        }
    }
}
