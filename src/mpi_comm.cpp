#include "mpi_comm.hpp"
#include <mpi.h>

static void exchange_variable(double **var, const DomainInfo& domain, int M_max) {
    double *send_left = new double[M_max + 1];
    double *send_right = new double[M_max + 1];
    double *recv_left = new double[M_max + 1];
    double *recv_right = new double[M_max + 1];
    
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

void exchange_ghost_cells_conservative(ConservativeVars& u0, const DomainInfo& domain, int M_max) {
    exchange_variable(u0.u_1, domain, M_max);
    exchange_variable(u0.u_2, domain, M_max);
    exchange_variable(u0.u_3, domain, M_max);
    exchange_variable(u0.u_4, domain, M_max);
    exchange_variable(u0.u_5, domain, M_max);
    exchange_variable(u0.u_6, domain, M_max);
    exchange_variable(u0.u_7, domain, M_max);
    exchange_variable(u0.u_8, domain, M_max);
}

void exchange_ghost_cells_physical(PhysicalFields& fields, const DomainInfo& domain, int M_max) {
    exchange_variable(fields.rho, domain, M_max);
    exchange_variable(fields.v_z, domain, M_max);
    exchange_variable(fields.v_r, domain, M_max);
    exchange_variable(fields.v_phi, domain, M_max);
    exchange_variable(fields.e, domain, M_max);
    exchange_variable(fields.p, domain, M_max);
    exchange_variable(fields.P, domain, M_max);
    exchange_variable(fields.H_z, domain, M_max);
    exchange_variable(fields.H_r, domain, M_max);
    exchange_variable(fields.H_phi, domain, M_max);
}
