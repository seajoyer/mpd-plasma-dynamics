#include "mpi_manager.hpp"

MPIManager::MPIManager(int& argc, char**& argv, const SimConfig& cfg) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    L_per_proc = cfg.L_max_global / size;

    l_start = rank * L_per_proc;
    l_end   = (rank + 1) * L_per_proc - 1;
    if (rank == size - 1)
        l_end = cfg.L_max_global - 1;   // last rank absorbs remainder

    local_L             = l_end - l_start + 1;
    local_L_with_ghosts = local_L + 2;
}

MPIManager::~MPIManager() {
    MPI_Finalize();
}

void MPIManager::exchange_ghosts(double** arr, int ncols,
                                  double* send_left,  double* send_right,
                                  double* recv_left,  double* recv_right) const {
    // Pack interior boundary slices
    for (int m = 0; m < ncols; ++m) {
        send_left[m]  = arr[1][m];          // leftmost interior cell
        send_right[m] = arr[local_L][m];    // rightmost interior cell
    }

    MPI_Request req[4];
    int nreq = 0;

    if (rank > 0) {
        MPI_Isend(send_left,  ncols, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &req[nreq++]);
        MPI_Irecv(recv_left,  ncols, MPI_DOUBLE, rank - 1, 1,
                  MPI_COMM_WORLD, &req[nreq++]);
    }
    if (rank < size - 1) {
        MPI_Isend(send_right, ncols, MPI_DOUBLE, rank + 1, 1,
                  MPI_COMM_WORLD, &req[nreq++]);
        MPI_Irecv(recv_right, ncols, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &req[nreq++]);
    }

    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

    // Unpack into ghost cells
    if (rank > 0)
        for (int m = 0; m < ncols; ++m) arr[0][m]             = recv_left[m];
    if (rank < size - 1)
        for (int m = 0; m < ncols; ++m) arr[local_L + 1][m]   = recv_right[m];
}
