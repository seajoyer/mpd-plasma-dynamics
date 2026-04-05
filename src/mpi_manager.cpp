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
// Single-array ghost exchange (kept for internal / legacy use)
// ============================================================

void MPIManager::exchange_ghosts(double** arr,
                                  double* col_sl, double* col_sr,
                                  double* col_rl, double* col_rr) const {
    const int nrows = local_L_with_ghosts;
    const int ncols = local_M_with_ghosts;

    if (nbr_m_lo != MPI_PROC_NULL)
        for (int l = 0; l < nrows; ++l) col_sl[l] = arr[l][1];
    if (nbr_m_hi != MPI_PROC_NULL)
        for (int l = 0; l < nrows; ++l) col_sr[l] = arr[l][local_M];

    MPI_Request reqs[8];
    int nreq = 0;

    if (nbr_l_lo != MPI_PROC_NULL) {
        MPI_Isend(arr[1],         ncols, MPI_DOUBLE, nbr_l_lo, 0, cart_comm, &reqs[nreq++]);
        MPI_Irecv(arr[0],         ncols, MPI_DOUBLE, nbr_l_lo, 1, cart_comm, &reqs[nreq++]);
    }
    if (nbr_l_hi != MPI_PROC_NULL) {
        MPI_Isend(arr[local_L],   ncols, MPI_DOUBLE, nbr_l_hi, 1, cart_comm, &reqs[nreq++]);
        MPI_Irecv(arr[local_L+1], ncols, MPI_DOUBLE, nbr_l_hi, 0, cart_comm, &reqs[nreq++]);
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

    if (nbr_m_lo != MPI_PROC_NULL)
        for (int l = 0; l < nrows; ++l) arr[l][0]         = col_rl[l];
    if (nbr_m_hi != MPI_PROC_NULL)
        for (int l = 0; l < nrows; ++l) arr[l][local_M+1] = col_rr[l];
}

// ============================================================
// Batched ghost exchange — all N arrays in one non-blocking round
// ============================================================
//
// Why this is faster than N sequential single-array exchanges:
//
//   Sequential:  for each array  { pack → 8×Isend/Irecv → Waitall → unpack }
//                = N serialised network round-trips.
//
//   Batched:     pack all → 8×N×Isend/Irecv → one Waitall → unpack all
//                = 1 network round-trip; the NIC can pipeline all transfers.
//
// Tag layout (must not collide with the single-array exchange tags 0-3):
//   Each array i uses tags  10 + i*4 + {0,1,2,3}
//   With n ≤ 18 arrays the maximum tag is 10 + 17*4 + 3 = 81, well within
//   any MPI implementation's MPI_TAG_UB (≥ 32767).
//
// col_bufs layout:
//   Segment for array i starts at  i * 4 * nrows.
//   Within the segment: [0..nrows-1]         = send_lo  (m-lo column)
//                       [nrows..2nrows-1]     = send_hi  (m-hi column)
//                       [2nrows..3nrows-1]    = recv_lo
//                       [3nrows..4nrows-1]    = recv_hi

void MPIManager::exchange_ghosts_batch(double** const* arrs, int n,
                                        std::vector<double>& col_bufs) const {
    const int nrows = local_L_with_ghosts;
    const int ncols = local_M_with_ghosts;

    col_bufs.resize(static_cast<std::size_t>(n) * 4 * nrows);

    // ---- Pack all m-direction send columns ----------------------------------
    for (int i = 0; i < n; ++i) {
        double* sl = col_bufs.data() + i * 4 * nrows + 0 * nrows;
        double* sr = col_bufs.data() + i * 4 * nrows + 1 * nrows;
        double** a = arrs[i];
        if (nbr_m_lo != MPI_PROC_NULL)
            for (int l = 0; l < nrows; ++l) sl[l] = a[l][1];
        if (nbr_m_hi != MPI_PROC_NULL)
            for (int l = 0; l < nrows; ++l) sr[l] = a[l][local_M];
    }

    // ---- Post all non-blocking sends and receives ---------------------------
    // Upper bound: 8 requests per array (4 directions × send+recv).
    std::vector<MPI_Request> reqs;
    reqs.reserve(static_cast<std::size_t>(n) * 8);

    for (int i = 0; i < n; ++i) {
        double** a  = arrs[i];
        const int t = 10 + i * 4;   // base tag for this array

        // L-direction: rows are contiguous — send/recv directly from the array.
        if (nbr_l_lo != MPI_PROC_NULL) {
            MPI_Request r0, r1;
            MPI_Isend(a[1],         ncols, MPI_DOUBLE, nbr_l_lo, t+0, cart_comm, &r0);
            MPI_Irecv(a[0],         ncols, MPI_DOUBLE, nbr_l_lo, t+1, cart_comm, &r1);
            reqs.push_back(r0); reqs.push_back(r1);
        }
        if (nbr_l_hi != MPI_PROC_NULL) {
            MPI_Request r0, r1;
            MPI_Isend(a[local_L],   ncols, MPI_DOUBLE, nbr_l_hi, t+1, cart_comm, &r0);
            MPI_Irecv(a[local_L+1], ncols, MPI_DOUBLE, nbr_l_hi, t+0, cart_comm, &r1);
            reqs.push_back(r0); reqs.push_back(r1);
        }

        // M-direction: use packed column buffers.
        double* sl = col_bufs.data() + i * 4 * nrows + 0 * nrows;
        double* sr = col_bufs.data() + i * 4 * nrows + 1 * nrows;
        double* rl = col_bufs.data() + i * 4 * nrows + 2 * nrows;
        double* rr = col_bufs.data() + i * 4 * nrows + 3 * nrows;

        if (nbr_m_lo != MPI_PROC_NULL) {
            MPI_Request r0, r1;
            MPI_Isend(sl, nrows, MPI_DOUBLE, nbr_m_lo, t+2, cart_comm, &r0);
            MPI_Irecv(rl, nrows, MPI_DOUBLE, nbr_m_lo, t+3, cart_comm, &r1);
            reqs.push_back(r0); reqs.push_back(r1);
        }
        if (nbr_m_hi != MPI_PROC_NULL) {
            MPI_Request r0, r1;
            MPI_Isend(sr, nrows, MPI_DOUBLE, nbr_m_hi, t+3, cart_comm, &r0);
            MPI_Irecv(rr, nrows, MPI_DOUBLE, nbr_m_hi, t+2, cart_comm, &r1);
            reqs.push_back(r0); reqs.push_back(r1);
        }
    }

    // ---- Wait for all transfers to complete ---------------------------------
    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    // ---- Unpack all m-direction receive columns ----------------------------
    for (int i = 0; i < n; ++i) {
        double** a  = arrs[i];
        double*  rl = col_bufs.data() + i * 4 * nrows + 2 * nrows;
        double*  rr = col_bufs.data() + i * 4 * nrows + 3 * nrows;
        if (nbr_m_lo != MPI_PROC_NULL)
            for (int l = 0; l < nrows; ++l) a[l][0]         = rl[l];
        if (nbr_m_hi != MPI_PROC_NULL)
            for (int l = 0; l < nrows; ++l) a[l][local_M+1] = rr[l];
    }
}
