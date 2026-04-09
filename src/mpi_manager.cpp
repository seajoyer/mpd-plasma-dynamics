#include "mpi_manager.hpp"

#include <cstring>

// ============================================================
// Constructor
// ============================================================

MPIManager::MPIManager(int& argc, char**& argv, const SimConfig& cfg) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Cartesian decomposition ----------------------------------------
    // Start from the config hints (0 = let MPI_Dims_create decide).
    dims[0] = cfg.mpi_dims_l;
    dims[1] = cfg.mpi_dims_m;
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0};
    // reorder=0 keeps WORLD rank == cart rank, simplifying IOManager.
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // ---- L decomposition (dimension 0) --------------------------------------
    L_per_proc = cfg.L_max / dims[0];
    l_start = coords[0] * L_per_proc;
    l_end = (coords[0] + 1) * L_per_proc - 1;
    if (coords[0] == dims[0] - 1) l_end = cfg.L_max - 1;
    local_L = l_end - l_start + 1;
    local_L_with_ghosts = local_L + 2;

    // ---- M decomposition (dimension 1) --------------------------------------
    M_per_proc = (cfg.M_max + 1) / dims[1];
    m_start = coords[1] * M_per_proc;
    m_end = (coords[1] + 1) * M_per_proc - 1;
    if (coords[1] == dims[1] - 1) m_end = cfg.M_max;
    local_M = m_end - m_start + 1;
    local_M_with_ghosts = local_M + 2;

    // ---- Neighbours ---------------------------------------------------------
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_l_lo, &nbr_l_hi);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_m_lo, &nbr_m_hi);
}

// ============================================================
// Destructor
// ============================================================

MPIManager::~MPIManager() {
    if (cart_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&cart_comm);
    }
    MPI_Finalize();
}

// ============================================================
// Single-array ghost exchange (kept for internal / legacy use)
// ============================================================

void MPIManager::ExchangeGhosts(double** arr, double* col_sl, double* col_sr,
                                double* col_rl, double* col_rr) const {
    const int nrows = local_L_with_ghosts;
    const int ncols = local_M_with_ghosts;

    if (nbr_m_lo != MPI_PROC_NULL) {
        for (int l = 0; l < nrows; ++l) col_sl[l] = arr[l][1];
    }
    if (nbr_m_hi != MPI_PROC_NULL) {
        for (int l = 0; l < nrows; ++l) col_sr[l] = arr[l][local_M];
    }

    MPI_Request reqs[8];
    int nreq = 0;

    if (nbr_l_lo != MPI_PROC_NULL) {
        MPI_Isend(arr[1], ncols, MPI_DOUBLE, nbr_l_lo, 0, cart_comm, &reqs[nreq++]);
        MPI_Irecv(arr[0], ncols, MPI_DOUBLE, nbr_l_lo, 1, cart_comm, &reqs[nreq++]);
    }
    if (nbr_l_hi != MPI_PROC_NULL) {
        MPI_Isend(arr[local_L], ncols, MPI_DOUBLE, nbr_l_hi, 1, cart_comm, &reqs[nreq++]);
        MPI_Irecv(arr[local_L + 1], ncols, MPI_DOUBLE, nbr_l_hi, 0, cart_comm,
                  &reqs[nreq++]);
    }
    if (nbr_m_lo != MPI_PROC_NULL) {
        MPI_Isend(col_sl, nrows, MPI_DOUBLE, nbr_m_lo, 2, cart_comm, &reqs[nreq++]);
        MPI_Irecv(col_rl, nrows, MPI_DOUBLE, nbr_m_lo, 3, cart_comm, &reqs[nreq++]);
    }
    if (nbr_m_hi != MPI_PROC_NULL) {
        MPI_Isend(col_sr, nrows, MPI_DOUBLE, nbr_m_hi, 3, cart_comm, &reqs[nreq++]);
        MPI_Irecv(col_rr, nrows, MPI_DOUBLE, nbr_m_hi, 2, cart_comm, &reqs[nreq++]);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    if (nbr_m_lo != MPI_PROC_NULL) {
        for (int l = 0; l < nrows; ++l) arr[l][0] = col_rl[l];
    }
    if (nbr_m_hi != MPI_PROC_NULL) {
        for (int l = 0; l < nrows; ++l) arr[l][local_M + 1] = col_rr[l];
    }
}

// ============================================================
// Batched ghost exchange — all N arrays, ONE message per direction
// ============================================================

void MPIManager::ExchangeGhostsBatch(double** const* arrs, int n,
                                     std::vector<double>& col_bufs) const {
    const int nrows = local_L_with_ghosts;
    const int ncols = local_M_with_ghosts;

    // Merged buffer: 4 row sections (each n*ncols) + 4 col sections (each n*nrows)
    const int row_seg = n * ncols;  // doubles per merged row direction message
    const int col_seg = n * nrows;  // doubles per merged col direction message
    col_bufs.resize(static_cast<std::size_t>(4) * row_seg +
                    static_cast<std::size_t>(4) * col_seg);

    double* row_sl = col_bufs.data();   // send to l-lo
    double* row_sr = row_sl + row_seg;  // send to l-hi
    double* row_rl = row_sr + row_seg;  // recv from l-lo
    double* row_rr = row_rl + row_seg;  // recv from l-hi
    double* col_sl = row_rr + row_seg;  // send to m-lo
    double* col_sr = col_sl + col_seg;  // send to m-hi
    double* col_rl = col_sr + col_seg;  // recv from m-lo
    double* col_rr = col_rl + col_seg;  // recv from m-hi

    // ---- Pack row sends (L-direction) using memcpy -------------------------
    // Array2D rows are contiguous so this is a fast bulk copy.
    for (int i = 0; i < n; ++i) {
        double** a = arrs[i];
        if (nbr_l_lo != MPI_PROC_NULL) {
            std::memcpy(row_sl + i * ncols, a[1], ncols * sizeof(double));
        }
        if (nbr_l_hi != MPI_PROC_NULL) {
            std::memcpy(row_sr + i * ncols, a[local_L], ncols * sizeof(double));
        }
    }

    // ---- Pack column sends (M-direction) -----------------------------------
    // With contiguous Array2D storage, a[l][1] = slab_base + l*ncols + 1.
    // Stride between elements is ncols — the CPU prefetcher handles this well.
    for (int i = 0; i < n; ++i) {
        double** a = arrs[i];
        double* sl = col_sl + i * nrows;
        double* sr = col_sr + i * nrows;
        if (nbr_m_lo != MPI_PROC_NULL) {
            for (int l = 0; l < nrows; ++l) sl[l] = a[l][1];
        }
        if (nbr_m_hi != MPI_PROC_NULL) {
            for (int l = 0; l < nrows; ++l) sr[l] = a[l][local_M];
        }
    }

    // ---- Post ONE Isend + ONE Irecv per active direction -------------------
    // Maximum 4 sends + 4 recvs = 8 MPI calls total (vs 72 in the old code).
    MPI_Request reqs[8];
    int nreq = 0;

    if (nbr_l_lo != MPI_PROC_NULL) {
        MPI_Isend(row_sl, row_seg, MPI_DOUBLE, nbr_l_lo, 0, cart_comm, &reqs[nreq++]);
        MPI_Irecv(row_rl, row_seg, MPI_DOUBLE, nbr_l_lo, 1, cart_comm, &reqs[nreq++]);
    }
    if (nbr_l_hi != MPI_PROC_NULL) {
        MPI_Isend(row_sr, row_seg, MPI_DOUBLE, nbr_l_hi, 1, cart_comm, &reqs[nreq++]);
        MPI_Irecv(row_rr, row_seg, MPI_DOUBLE, nbr_l_hi, 0, cart_comm, &reqs[nreq++]);
    }
    if (nbr_m_lo != MPI_PROC_NULL) {
        MPI_Isend(col_sl, col_seg, MPI_DOUBLE, nbr_m_lo, 2, cart_comm, &reqs[nreq++]);
        MPI_Irecv(col_rl, col_seg, MPI_DOUBLE, nbr_m_lo, 3, cart_comm, &reqs[nreq++]);
    }
    if (nbr_m_hi != MPI_PROC_NULL) {
        MPI_Isend(col_sr, col_seg, MPI_DOUBLE, nbr_m_hi, 3, cart_comm, &reqs[nreq++]);
        MPI_Irecv(col_rr, col_seg, MPI_DOUBLE, nbr_m_hi, 2, cart_comm, &reqs[nreq++]);
    }

    if (nreq > 0) {
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
    }

    // ---- Unpack row receives (L-direction) using memcpy --------------------
    for (int i = 0; i < n; ++i) {
        double** a = arrs[i];
        if (nbr_l_lo != MPI_PROC_NULL) {
            std::memcpy(a[0], row_rl + i * ncols, ncols * sizeof(double));
        }
        if (nbr_l_hi != MPI_PROC_NULL) {
            std::memcpy(a[local_L + 1], row_rr + i * ncols, ncols * sizeof(double));
        }
    }

    // ---- Unpack column receives (M-direction) ------------------------------
    for (int i = 0; i < n; ++i) {
        double** a = arrs[i];
        double* rl = col_rl + i * nrows;
        double* rr = col_rr + i * nrows;
        if (nbr_m_lo != MPI_PROC_NULL) {
            for (int l = 0; l < nrows; ++l) a[l][0] = rl[l];
        }
        if (nbr_m_hi != MPI_PROC_NULL) {
            for (int l = 0; l < nrows; ++l) a[l][local_M + 1] = rr[l];
        }
    }
}
