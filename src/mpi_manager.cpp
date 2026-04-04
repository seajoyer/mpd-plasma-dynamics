#include "mpi_manager.hpp"

// ============================================================
// Constructor
// ============================================================

MPIManager::MPIManager(int& argc, char**& argv, const SimConfig& cfg) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- 2-D Cartesian decomposition ----------------------------------------
    dims[0] = dims[1] = 0;
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0};
    // reorder=0 keeps WORLD rank == cart rank, simplifying IOManager.
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // ---- L decomposition (dimension 0) --------------------------------------
    L_per_proc = cfg.L_max_global / dims[0];
    l_start    = coords[0] * L_per_proc;
    l_end      = (coords[0] + 1) * L_per_proc - 1;
    if (coords[0] == dims[0] - 1) l_end = cfg.L_max_global - 1;
    local_L             = l_end - l_start + 1;
    local_L_with_ghosts = local_L + 2;

    // ---- M decomposition (dimension 1) --------------------------------------
    // Total M nodes: M_max+1, indices 0..M_max.
    // Bottom rank (coords[1]==0) owns global m=0 at m_local=1.
    // Top    rank (coords[1]==dims[1]-1) owns global m=M_max at m_local=local_M.
    M_per_proc = (cfg.M_max + 1) / dims[1];
    m_start    = coords[1] * M_per_proc;
    m_end      = (coords[1] + 1) * M_per_proc - 1;
    if (coords[1] == dims[1] - 1) m_end = cfg.M_max;
    local_M             = m_end - m_start + 1;
    local_M_with_ghosts = local_M + 2;

    // ---- Neighbours ---------------------------------------------------------
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_l_lo, &nbr_l_hi);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_m_lo, &nbr_m_hi);
}

// ============================================================
// Destructor
// ============================================================

MPIManager::~MPIManager() {
    if (cart_comm != MPI_COMM_NULL)
        MPI_Comm_free(&cart_comm);
    MPI_Finalize();
}

// ============================================================
// Ghost-cell exchange — all four directions, one Waitall
// ============================================================

void MPIManager::exchange_ghosts(double** arr,
                                  double* row_sl, double* row_sr,
                                  double* row_rl, double* row_rr,
                                  double* col_sl, double* col_sr,
                                  double* col_rl, double* col_rr) const {
    const int nrows = local_L_with_ghosts;  // local_L + 2
    const int ncols = local_M_with_ghosts;  // local_M + 2

    // Pack column send buffers before launching any Isend.
    // (Row send buffers point directly into arr[1] / arr[local_L].)
    if (nbr_m_lo != MPI_PROC_NULL)
        for (int l = 0; l < nrows; ++l) col_sl[l] = arr[l][1];
    if (nbr_m_hi != MPI_PROC_NULL)
        for (int l = 0; l < nrows; ++l) col_sr[l] = arr[l][local_M];

    MPI_Request reqs[8];
    int nreq = 0;

    // L-direction — rows are contiguous in memory.
    // Tag convention: 0 = l_lo→l_hi data,  1 = l_hi→l_lo data.
    if (nbr_l_lo != MPI_PROC_NULL) {
        MPI_Isend(arr[1],         ncols, MPI_DOUBLE, nbr_l_lo, 0, cart_comm, &reqs[nreq++]);
        MPI_Irecv(arr[0],         ncols, MPI_DOUBLE, nbr_l_lo, 1, cart_comm, &reqs[nreq++]);
    }
    if (nbr_l_hi != MPI_PROC_NULL) {
        MPI_Isend(arr[local_L],   ncols, MPI_DOUBLE, nbr_l_hi, 1, cart_comm, &reqs[nreq++]);
        MPI_Irecv(arr[local_L+1], ncols, MPI_DOUBLE, nbr_l_hi, 0, cart_comm, &reqs[nreq++]);
    }

    // M-direction — columns are packed into col_s* buffers above.
    // Tag convention: 2 = m_lo→m_hi data,  3 = m_hi→m_lo data.
    if (nbr_m_lo != MPI_PROC_NULL) {
        MPI_Isend(col_sl, nrows, MPI_DOUBLE, nbr_m_lo, 2, cart_comm, &reqs[nreq++]);
        MPI_Irecv(col_rl, nrows, MPI_DOUBLE, nbr_m_lo, 3, cart_comm, &reqs[nreq++]);
    }
    if (nbr_m_hi != MPI_PROC_NULL) {
        MPI_Isend(col_sr, nrows, MPI_DOUBLE, nbr_m_hi, 3, cart_comm, &reqs[nreq++]);
        MPI_Irecv(col_rr, nrows, MPI_DOUBLE, nbr_m_hi, 2, cart_comm, &reqs[nreq++]);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Unpack column receives into ghost columns.
    if (nbr_m_lo != MPI_PROC_NULL)
        for (int l = 0; l < nrows; ++l) arr[l][0]         = col_rl[l];
    if (nbr_m_hi != MPI_PROC_NULL)
        for (int l = 0; l < nrows; ++l) arr[l][local_M+1] = col_rr[l];
}
