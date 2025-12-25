#include "mpi_comm.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

// ============================================================================
// 2D Domain Decomposition Setup
// ============================================================================

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

void Setup2DDecomposition(DomainInfo& domain, const SimulationParams& params) {
    // Get basic MPI info
    MPI_Comm_rank(MPI_COMM_WORLD, &domain.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &domain.size);
    
    // Compute optimal grid dimensions
    ComputeOptimalDims(domain.size, params.L_max_global, params.M_max + 1, domain.dims);
    
    // Create 2D Cartesian topology
    // periods = {0, 0} - no periodic boundaries
    int periods[2] = {0, 0};
    int reorder = 1;  // Allow MPI to reorder ranks for efficiency
    
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, domain.dims, periods, reorder, &cart_comm);
    domain.cart_comm = static_cast<int>(cart_comm);
    
    // Get this process's coordinates in the grid
    MPI_Cart_coords(cart_comm, domain.rank, 2, domain.coords);
    MPI_Comm_rank(cart_comm, &domain.cart_rank);
    
    // Get neighbor ranks
    // L direction (dimension 0): left = -1, right = +1
    MPI_Cart_shift(cart_comm, 0, 1, &domain.neighbor_left, &domain.neighbor_right);
    // M direction (dimension 1): down = -1, up = +1
    MPI_Cart_shift(cart_comm, 1, 1, &domain.neighbor_down, &domain.neighbor_up);
    
    // Set boundary flags
    domain.is_left_boundary = (domain.coords[0] == 0);
    domain.is_right_boundary = (domain.coords[0] == domain.dims[0] - 1);
    domain.is_down_boundary = (domain.coords[1] == 0);
    domain.is_up_boundary = (domain.coords[1] == domain.dims[1] - 1);
    
    // Compute L-direction decomposition
    domain.L_per_proc = params.L_max_global / domain.dims[0];
    int L_remainder = params.L_max_global % domain.dims[0];
    
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
    int M_total = params.M_max + 1;  // M goes from 0 to M_max inclusive
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

void PrintDecompositionInfo(const DomainInfo& domain, const SimulationParams& params) {
    MPI_Comm cart_comm = GetCartComm(domain);
    
    // Print from each rank in order
    for (int r = 0; r < domain.size; r++) {
        if (domain.rank == r) {
            printf("Rank %d: coords=(%d,%d), L=[%d,%d] (%d cells), M=[%d,%d] (%d cells)\n",
                   domain.rank, domain.coords[0], domain.coords[1],
                   domain.l_start, domain.l_end, domain.local_L,
                   domain.m_start, domain.m_end, domain.local_M);
            printf("        neighbors: left=%d, right=%d, down=%d, up=%d\n",
                   domain.neighbor_left, domain.neighbor_right,
                   domain.neighbor_down, domain.neighbor_up);
            printf("        boundaries: left=%d, right=%d, down=%d, up=%d\n",
                   domain.is_left_boundary, domain.is_right_boundary,
                   domain.is_down_boundary, domain.is_up_boundary);
            fflush(stdout);
        }
        MPI_Barrier(cart_comm);
    }
    
    if (domain.rank == 0) {
        printf("\n2D Decomposition: %d x %d processes for %d x %d domain\n",
               domain.dims[0], domain.dims[1], params.L_max_global, params.M_max + 1);
    }
}

// ============================================================================
// Ghost Cell Exchange - 2D Version
// ============================================================================

// Helper: Exchange a single 2D variable in L-direction (send/recv columns)
static void ExchangeVariableL(double** var, const DomainInfo& domain) {
    MPI_Comm cart_comm = GetCartComm(domain);
    const int local_L = domain.local_L;
    const int local_M_with_ghosts = domain.local_M_with_ghosts;
    
    // Allocate buffers for column exchange
    std::vector<double> send_left(local_M_with_ghosts);
    std::vector<double> send_right(local_M_with_ghosts);
    std::vector<double> recv_left(local_M_with_ghosts);
    std::vector<double> recv_right(local_M_with_ghosts);
    
    // Pack columns to send
    for (int m = 0; m < local_M_with_ghosts; m++) {
        send_left[m] = var[1][m];           // First interior column
        send_right[m] = var[local_L][m];    // Last interior column
    }
    
    MPI_Request req[4];
    int req_count = 0;
    
    // Exchange with left neighbor
    if (domain.neighbor_left >= 0) {
        MPI_Isend(send_left.data(), local_M_with_ghosts, MPI_DOUBLE,
                  domain.neighbor_left, 0, cart_comm, &req[req_count++]);
        MPI_Irecv(recv_left.data(), local_M_with_ghosts, MPI_DOUBLE,
                  domain.neighbor_left, 1, cart_comm, &req[req_count++]);
    }
    
    // Exchange with right neighbor
    if (domain.neighbor_right >= 0) {
        MPI_Isend(send_right.data(), local_M_with_ghosts, MPI_DOUBLE,
                  domain.neighbor_right, 1, cart_comm, &req[req_count++]);
        MPI_Irecv(recv_right.data(), local_M_with_ghosts, MPI_DOUBLE,
                  domain.neighbor_right, 0, cart_comm, &req[req_count++]);
    }
    
    MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
    
    // Unpack received columns to ghost cells
    if (domain.neighbor_left >= 0) {
        for (int m = 0; m < local_M_with_ghosts; m++) {
            var[0][m] = recv_left[m];
        }
    }
    if (domain.neighbor_right >= 0) {
        for (int m = 0; m < local_M_with_ghosts; m++) {
            var[local_L + 1][m] = recv_right[m];
        }
    }
}

// Helper: Exchange a single 2D variable in M-direction (send/recv rows)
static void ExchangeVariableM(double** var, const DomainInfo& domain) {
    MPI_Comm cart_comm = GetCartComm(domain);
    const int local_L_with_ghosts = domain.local_L_with_ghosts;
    const int local_M = domain.local_M;
    
    // Allocate buffers for row exchange
    std::vector<double> send_down(local_L_with_ghosts);
    std::vector<double> send_up(local_L_with_ghosts);
    std::vector<double> recv_down(local_L_with_ghosts);
    std::vector<double> recv_up(local_L_with_ghosts);
    
    // Pack rows to send
    for (int l = 0; l < local_L_with_ghosts; l++) {
        send_down[l] = var[l][1];           // First interior row
        send_up[l] = var[l][local_M];       // Last interior row
    }
    
    MPI_Request req[4];
    int req_count = 0;
    
    // Exchange with down neighbor (lower M)
    if (domain.neighbor_down >= 0) {
        MPI_Isend(send_down.data(), local_L_with_ghosts, MPI_DOUBLE,
                  domain.neighbor_down, 2, cart_comm, &req[req_count++]);
        MPI_Irecv(recv_down.data(), local_L_with_ghosts, MPI_DOUBLE,
                  domain.neighbor_down, 3, cart_comm, &req[req_count++]);
    }
    
    // Exchange with up neighbor (higher M)
    if (domain.neighbor_up >= 0) {
        MPI_Isend(send_up.data(), local_L_with_ghosts, MPI_DOUBLE,
                  domain.neighbor_up, 3, cart_comm, &req[req_count++]);
        MPI_Irecv(recv_up.data(), local_L_with_ghosts, MPI_DOUBLE,
                  domain.neighbor_up, 2, cart_comm, &req[req_count++]);
    }
    
    MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
    
    // Unpack received rows to ghost cells
    if (domain.neighbor_down >= 0) {
        for (int l = 0; l < local_L_with_ghosts; l++) {
            var[l][0] = recv_down[l];
        }
    }
    if (domain.neighbor_up >= 0) {
        for (int l = 0; l < local_L_with_ghosts; l++) {
            var[l][local_M + 1] = recv_up[l];
        }
    }
}

void ExchangeGhostCellsConservative2D(ConservativeVars& u0, const DomainInfo& domain,
                                       const SimulationParams& params) {
    // Exchange in L direction first
    ExchangeVariableL(u0.u_1, domain);
    ExchangeVariableL(u0.u_2, domain);
    ExchangeVariableL(u0.u_3, domain);
    ExchangeVariableL(u0.u_4, domain);
    ExchangeVariableL(u0.u_5, domain);
    ExchangeVariableL(u0.u_6, domain);
    ExchangeVariableL(u0.u_7, domain);
    ExchangeVariableL(u0.u_8, domain);
    
    // Then exchange in M direction (includes corners from L exchange)
    ExchangeVariableM(u0.u_1, domain);
    ExchangeVariableM(u0.u_2, domain);
    ExchangeVariableM(u0.u_3, domain);
    ExchangeVariableM(u0.u_4, domain);
    ExchangeVariableM(u0.u_5, domain);
    ExchangeVariableM(u0.u_6, domain);
    ExchangeVariableM(u0.u_7, domain);
    ExchangeVariableM(u0.u_8, domain);
}

void ExchangeGhostCellsPhysical2D(PhysicalFields& fields, const DomainInfo& domain,
                                   const SimulationParams& params) {
    // Exchange in L direction first
    ExchangeVariableL(fields.rho, domain);
    ExchangeVariableL(fields.v_z, domain);
    ExchangeVariableL(fields.v_r, domain);
    ExchangeVariableL(fields.v_phi, domain);
    ExchangeVariableL(fields.e, domain);
    ExchangeVariableL(fields.p, domain);
    ExchangeVariableL(fields.P, domain);
    ExchangeVariableL(fields.H_z, domain);
    ExchangeVariableL(fields.H_r, domain);
    ExchangeVariableL(fields.H_phi, domain);
    
    // Then exchange in M direction
    ExchangeVariableM(fields.rho, domain);
    ExchangeVariableM(fields.v_z, domain);
    ExchangeVariableM(fields.v_r, domain);
    ExchangeVariableM(fields.v_phi, domain);
    ExchangeVariableM(fields.e, domain);
    ExchangeVariableM(fields.p, domain);
    ExchangeVariableM(fields.P, domain);
    ExchangeVariableM(fields.H_z, domain);
    ExchangeVariableM(fields.H_r, domain);
    ExchangeVariableM(fields.H_phi, domain);
}

// ============================================================================
// Data Gathering for Output - 2D Version
// ============================================================================

void GatherResultsToRank0_2D(const PhysicalFields& fields, const GridGeometry& grid,
                              const DomainInfo& domain, const SimulationParams& params,
                              PhysicalFields& global_fields, GridGeometry& global_grid) {
    MPI_Comm cart_comm = GetCartComm(domain);
    const int L_max_global = params.L_max_global;
    const int M_max = params.M_max;
    
    // Create subcommunicators for rows (same L coord) and columns (same M coord)
    MPI_Comm row_comm, col_comm;
    
    // Row communicator: processes with same coords[0] (L position)
    MPI_Comm_split(cart_comm, domain.coords[0], domain.coords[1], &row_comm);
    // Column communicator: processes with same coords[1] (M position)
    MPI_Comm_split(cart_comm, domain.coords[1], domain.coords[0], &col_comm);
    
    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);
    
    // Gather sizes for this row (M direction)
    std::vector<int> m_counts(domain.dims[1]);
    std::vector<int> m_displs(domain.dims[1]);
    MPI_Allgather(&domain.local_M, 1, MPI_INT, m_counts.data(), 1, MPI_INT, row_comm);
    
    m_displs[0] = 0;
    for (int i = 1; i < domain.dims[1]; i++) {
        m_displs[i] = m_displs[i-1] + m_counts[i-1];
    }
    int M_total = m_displs[domain.dims[1]-1] + m_counts[domain.dims[1]-1];
    
    // Gather sizes for this column (L direction)
    std::vector<int> l_counts(domain.dims[0]);
    std::vector<int> l_displs(domain.dims[0]);
    MPI_Allgather(&domain.local_L, 1, MPI_INT, l_counts.data(), 1, MPI_INT, col_comm);
    
    l_displs[0] = 0;
    for (int i = 1; i < domain.dims[0]; i++) {
        l_displs[i] = l_displs[i-1] + l_counts[i-1];
    }
    
    // Process each field - gather row by row in M, then gather to rank 0 in L
    auto gatherField = [&](double** local_field, double** global_field) {
        // For each local L index, gather the M direction first within the row
        for (int l_local = 0; l_local < domain.local_L; l_local++) {
            // Extract local M data (excluding ghosts)
            std::vector<double> local_m_data(domain.local_M);
            for (int m = 0; m < domain.local_M; m++) {
                local_m_data[m] = local_field[l_local + 1][m + 1];  // Skip ghost cells
            }
            
            // Gather full M row on row_rank 0
            std::vector<double> full_m_row;
            if (row_rank == 0) {
                full_m_row.resize(M_total);
            }
            
            MPI_Gatherv(local_m_data.data(), domain.local_M, MPI_DOUBLE,
                        full_m_row.data(), m_counts.data(), m_displs.data(),
                        MPI_DOUBLE, 0, row_comm);
            
            // Now gather these rows to rank 0 in the column communicator
            if (row_rank == 0) {
                int l_global = domain.l_start + l_local;
                
                // Gather to col_rank 0 (which is global rank 0)
                if (col_rank == 0 && domain.rank == 0) {
                    // We're rank 0 - receive from all column members
                    // First, copy our own data
                    for (int m = 0; m < M_total; m++) {
                        global_field[l_global][m] = full_m_row[m];
                    }
                    
                    // Receive from other ranks in column
                    for (int src = 1; src < domain.dims[0]; src++) {
                        int src_l_start = l_displs[src];
                        int src_l_count = l_counts[src];
                        
                        for (int ll = 0; ll < src_l_count; ll++) {
                            std::vector<double> recv_row(M_total);
                            MPI_Recv(recv_row.data(), M_total, MPI_DOUBLE,
                                     src, 100 + ll, col_comm, MPI_STATUS_IGNORE);
                            for (int m = 0; m < M_total; m++) {
                                global_field[src_l_start + ll][m] = recv_row[m];
                            }
                        }
                    }
                } else if (col_rank != 0) {
                    // Send our row to rank 0 in column
                    MPI_Send(full_m_row.data(), M_total, MPI_DOUBLE,
                             0, 100 + l_local, col_comm);
                }
            }
        }
    };
    
    // Gather all fields
    gatherField(fields.rho, global_fields.rho);
    gatherField(fields.v_z, global_fields.v_z);
    gatherField(fields.v_r, global_fields.v_r);
    gatherField(fields.v_phi, global_fields.v_phi);
    gatherField(fields.e, global_fields.e);
    gatherField(fields.H_z, global_fields.H_z);
    gatherField(fields.H_r, global_fields.H_r);
    gatherField(fields.H_phi, global_fields.H_phi);
    gatherField(grid.r, global_grid.r);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

// ============================================================================
// Legacy 1D functions (kept for compatibility)
// ============================================================================

void GatherResultsToRank0(const PhysicalFields& fields, const GridGeometry& grid,
                          const DomainInfo& domain, const SimulationParams& params,
                          PhysicalFields& global_fields, GridGeometry& global_grid) {
    const int M_max = params.M_max;
    const int L_max_global = params.L_max_global;
    const int local_L = domain.local_L;
    const int L_per_proc = domain.L_per_proc;

    // Gather data row by row (excluding ghost cells)
    for (int m = 0; m < M_max + 1; m++) {
        auto* local_row_rho = new double[local_L];
        auto* local_row_vz = new double[local_L];
        auto* local_row_vr = new double[local_L];
        auto* local_row_vphi = new double[local_L];
        auto* local_row_e = new double[local_L];
        auto* local_row_Hz = new double[local_L];
        auto* local_row_Hr = new double[local_L];
        auto* local_row_Hphi = new double[local_L];
        auto* local_row_r = new double[local_L];

        for (int l = 0; l < local_L; l++) {
            local_row_rho[l] = fields.rho[l + 1][m];
            local_row_vz[l] = fields.v_z[l + 1][m];
            local_row_vr[l] = fields.v_r[l + 1][m];
            local_row_vphi[l] = fields.v_phi[l + 1][m];
            local_row_e[l] = fields.e[l + 1][m];
            local_row_Hz[l] = fields.H_z[l + 1][m];
            local_row_Hr[l] = fields.H_r[l + 1][m];
            local_row_Hphi[l] = fields.H_phi[l + 1][m];
            local_row_r[l] = grid.r[l + 1][m];
        }

        double *global_row_rho = nullptr, *global_row_vz = nullptr,
               *global_row_vr = nullptr;
        double *global_row_vphi = nullptr, *global_row_e = nullptr;
        double *global_row_Hz = nullptr, *global_row_Hr = nullptr,
               *global_row_Hphi = nullptr;
        double* global_row_r = nullptr;

        if (domain.rank == 0) {
            global_row_rho = new double[L_max_global];
            global_row_vz = new double[L_max_global];
            global_row_vr = new double[L_max_global];
            global_row_vphi = new double[L_max_global];
            global_row_e = new double[L_max_global];
            global_row_Hz = new double[L_max_global];
            global_row_Hr = new double[L_max_global];
            global_row_Hphi = new double[L_max_global];
            global_row_r = new double[L_max_global];
        }

        int* recvcounts = new int[domain.size];
        int* displs = new int[domain.size];

        for (int i = 0; i < domain.size; i++) {
            int i_start = i * L_per_proc;
            int i_end = (i + 1) * L_per_proc - 1;
            if (i == domain.size - 1) i_end = L_max_global - 1;
            recvcounts[i] = i_end - i_start + 1;
            displs[i] = i_start;
        }

        MPI_Gatherv(local_row_rho, local_L, MPI_DOUBLE, global_row_rho, recvcounts,
                    displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_vz, local_L, MPI_DOUBLE, global_row_vz, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_vr, local_L, MPI_DOUBLE, global_row_vr, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_vphi, local_L, MPI_DOUBLE, global_row_vphi, recvcounts,
                    displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_e, local_L, MPI_DOUBLE, global_row_e, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_Hz, local_L, MPI_DOUBLE, global_row_Hz, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_Hr, local_L, MPI_DOUBLE, global_row_Hr, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_Hphi, local_L, MPI_DOUBLE, global_row_Hphi, recvcounts,
                    displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_row_r, local_L, MPI_DOUBLE, global_row_r, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (domain.rank == 0) {
            for (int l = 0; l < L_max_global; l++) {
                global_fields.rho[l][m] = global_row_rho[l];
                global_fields.v_z[l][m] = global_row_vz[l];
                global_fields.v_r[l][m] = global_row_vr[l];
                global_fields.v_phi[l][m] = global_row_vphi[l];
                global_fields.e[l][m] = global_row_e[l];
                global_fields.H_z[l][m] = global_row_Hz[l];
                global_fields.H_r[l][m] = global_row_Hr[l];
                global_fields.H_phi[l][m] = global_row_Hphi[l];
                global_grid.r[l][m] = global_row_r[l];
            }

            delete[] global_row_rho;
            delete[] global_row_vz;
            delete[] global_row_vr;
            delete[] global_row_vphi;
            delete[] global_row_e;
            delete[] global_row_Hz;
            delete[] global_row_Hr;
            delete[] global_row_Hphi;
            delete[] global_row_r;
        }

        delete[] local_row_rho;
        delete[] local_row_vz;
        delete[] local_row_vr;
        delete[] local_row_vphi;
        delete[] local_row_e;
        delete[] local_row_Hz;
        delete[] local_row_Hr;
        delete[] local_row_Hphi;
        delete[] local_row_r;
        delete[] recvcounts;
        delete[] displs;
    }
}

static void ExchangeVariable(double **var, const DomainInfo& domain, int M_max) {
    auto *send_left = new double[M_max + 1];
    auto *send_right = new double[M_max + 1];
    auto *recv_left = new double[M_max + 1];
    auto *recv_right = new double[M_max + 1];
    
    for (int m = 0; m < M_max + 1; m++) {
        send_left[m] = var[1][m];
        send_right[m] = var[domain.local_L][m];
    }
    
    MPI_Request req[4];
    int req_count = 0;
    
    // Send to left, receive from left
    if (domain.rank > 0) {
        MPI_Isend(send_left, M_max + 1, MPI_DOUBLE, domain.rank - 1, 0, MPI_COMM_WORLD, &req[req_count++]);
        MPI_Irecv(recv_left, M_max + 1, MPI_DOUBLE, domain.rank - 1, 1, MPI_COMM_WORLD, &req[req_count++]);
    }
    
    // Send to right, receive from right
    if (domain.rank < domain.size - 1) {
        MPI_Isend(send_right, M_max + 1, MPI_DOUBLE, domain.rank + 1, 1, MPI_COMM_WORLD, &req[req_count++]);
        MPI_Irecv(recv_right, M_max + 1, MPI_DOUBLE, domain.rank + 1, 0, MPI_COMM_WORLD, &req[req_count++]);
    }
    
    MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
    
    // Copy received data to ghost cells
    if (domain.rank > 0) {
        for (int m = 0; m < M_max + 1; m++) {
            var[0][m] = recv_left[m];
        }
    }
    if (domain.rank < domain.size - 1) {
        for (int m = 0; m < M_max + 1; m++) {
            var[domain.local_L + 1][m] = recv_right[m];
        }
    }
    
    delete[] send_left;
    delete[] send_right;
    delete[] recv_left;
    delete[] recv_right;
}

void ExchangeGhostCellsConservative(ConservativeVars& u0, const DomainInfo& domain, int M_max) {
    ExchangeVariable(u0.u_1, domain, M_max);
    ExchangeVariable(u0.u_2, domain, M_max);
    ExchangeVariable(u0.u_3, domain, M_max);
    ExchangeVariable(u0.u_4, domain, M_max);
    ExchangeVariable(u0.u_5, domain, M_max);
    ExchangeVariable(u0.u_6, domain, M_max);
    ExchangeVariable(u0.u_7, domain, M_max);
    ExchangeVariable(u0.u_8, domain, M_max);
}

void ExchangeGhostCellsPhysical(PhysicalFields& fields, const DomainInfo& domain, int M_max) {
    ExchangeVariable(fields.rho, domain, M_max);
    ExchangeVariable(fields.v_z, domain, M_max);
    ExchangeVariable(fields.v_r, domain, M_max);
    ExchangeVariable(fields.v_phi, domain, M_max);
    ExchangeVariable(fields.e, domain, M_max);
    ExchangeVariable(fields.p, domain, M_max);
    ExchangeVariable(fields.P, domain, M_max);
    ExchangeVariable(fields.H_z, domain, M_max);
    ExchangeVariable(fields.H_r, domain, M_max);
    ExchangeVariable(fields.H_phi, domain, M_max);
}
