#include "mpi_comm.hpp"
#include <mpi.h>

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
