#include<stdio.h>
#include<iostream>
#include<fstream>
#include<cmath>
#include<omp.h>
#include<mpi.h>

void memory_allocation_2D(double** &array, int rows, int columns) {
	array = new double* [rows];
    for (int i = 0; i < rows; i++) {
        array[i] = new double[columns];
        for (int j = 0; j < columns; j++) {
            array[i][j] = 0;
        }
    }
}

void memory_clearing_2D(double** &array, int rows) {
	for (int i = 0; i < rows; i++) {
		delete [] array[i];
	}
	delete [] array;
}

double r1(double z) {
	if (z < 0.3) {
		return 0.2;
	} else if (z >=0.3 && z < 0.4) {
		return 0.2 - 10 * pow((z - 0.3), 2);
	} else if (z >=0.4 && z < 0.478) {
		return 10 * pow((z - 0.5), 2);
	} else {
		return 0.005;
	}
}

double r2(double z) {
	return 0.8;
}

double der_r1(double z) {
	if (z < 0.3) {
		return 0;
	} else if (z >=0.3 && z < 0.4) {
		return - 10 * 2 * (z - 0.3);
	} else if (z >=0.4 && z < 0.478) {
		return 10 * 2 * (z - 0.5);
	} else {
		return 0;
	}
}

double der_r2(double z) {
	return 0;
}

void animate_write(int n, int L_max, int M_max, double dz, double **r, 
				   double **rho, double **v_z, double **v_r, double **v_phi,
				   double **e, double **H_z, double **H_r, double **H_phi,
				   int rank, int size, int l_start_global) {

	char filename[50];
    sprintf(filename, "animate_m_29_800x400/%d_rank%d.plt", n, rank);

	std :: ofstream out(filename);
	int local_L = L_max;
	int np = (local_L + 1) * (M_max + 1);
	int ne = local_L * M_max;
	double hfr;
	out << "VARIABLES=\n";
	out << "\"r\"\n\"z\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n\"Energy\"\n\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
	out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne << "\n ";
	for (int m = 0; m < M_max + 1; m++) {
		for (int l = 0; l < local_L + 1; l++) {
			hfr = H_phi[l][m] * r[l][m];
			out << (l_start_global + l) * dz << " " << r[l][m] << " " << rho[l][m] << " " << 
				v_z[l][m] << " " << v_r[l][m] << " " << std::sqrt(v_z[l][m] * v_z[l][m] + v_r[l][m] * v_r[l][m]) << " " << 
				v_phi[l][m] << " " << e[l][m] << " " << H_z[l][m] << " " << H_r[l][m] << " " << hfr << " " << H_phi[l][m] << "\n";
		}
	}

	int i1 = 0;
	int i2 = 0;
	int i3 = 0;
	int i4 = 0;
	
	for (int m = 0; m < M_max; m++) {
		for (int l = 0; l < local_L; l++) {
			i1 = l + m * (local_L + 1) + 1;
			i2 = l + 1 + m * (local_L + 1) + 1;
			i3 = l + 1 + (m + 1) * (local_L + 1) + 1;
			i4 = l + (m + 1) * (local_L + 1) + 1;
			out << i1 << " " << i2 << " " << i3 << " " << i4 << "\n";
		}
	}

	out.close();
}

double max_array(double **array, double L, double M) {
	double maxim = 0.0;

	for (int l = 0; l < L + 1; l++) {
		for (int m = 0; m < M + 1; m++) {
			if (maxim < array[l][m]) {
				maxim = array[l][m];
			}
		}
	}

	return maxim;
}

int main(int argc, char* argv[]) {

	// Initialize MPI
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// task data
    double gamma = 1.67;
	double beta = 0.05;
	double H_z0 = 0.25;
	int animate = 0;

    // discrete solution area
    double T = 10.0;
    double t = 0.0;
    double dt = 0.000025;

    int L_max_global = 800;
	int L_end = 320;
	int M_max = 400;

	double dz = 1.0 / L_max_global;
	double dy = 1.0 / M_max;

	// Domain decomposition in z-direction (l-direction)
	int L_per_proc = L_max_global / size;
	int L_remainder = L_max_global % size;
	
	// Each process gets L_per_proc cells, last process gets remainder
	int l_start = rank * L_per_proc;
	int l_end = (rank + 1) * L_per_proc - 1;
	if (rank == size - 1) {
		l_end = L_max_global - 1;
	}
	
	int local_L = l_end - l_start + 1;
	
	// Add ghost cells for boundary exchange
	int local_L_with_ghosts = local_L + 2;

	// parallel parameters
	int procs = 1;
	if (argc > 1) {
		procs = atoi(argv[1]);
	}

	omp_set_num_threads(procs);

	double begin, end, total;

	// creating arrays for u components (local with ghost cells)
	double **u_1, **u_2, **u_3, **u_4, **u_5, **u_6, **u_7, **u_8;
	double **u0_1, **u0_2, **u0_3, **u0_4, **u0_5, **u0_6, **u0_7, **u0_8;
	
	memory_allocation_2D(u_1, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u_2, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u_3, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u_4, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u_5, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u_6, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u_7, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u_8, local_L_with_ghosts, M_max + 1);

	memory_allocation_2D(u0_1, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u0_2, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u0_3, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u0_4, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u0_5, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u0_6, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u0_7, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(u0_8, local_L_with_ghosts, M_max + 1);

	// creating arrays for physical parameters
	double **rho, **v_r, **v_phi, **v_z, **e, **p, **P;
	double **H_r, **H_phi, **H_z;
	
	memory_allocation_2D(rho, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(v_r, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(v_phi, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(v_z, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(e, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(p, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(P, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(H_r, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(H_phi, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(H_z, local_L_with_ghosts, M_max + 1);

	double **r, **r_z;
	memory_allocation_2D(r, local_L_with_ghosts, M_max + 1);
	memory_allocation_2D(r_z, local_L_with_ghosts, M_max + 1);
	double *R = new double [local_L_with_ghosts];
	double *dr = new double [local_L_with_ghosts];

	// filling zeros
	for (int l = 0; l < local_L_with_ghosts; l++) {
		R[l] = 0;
		dr[l] = 0;
	}

	// creating arrays for axes (using global indices)
	double r_0 = (r1(0) + r2(0)) / 2.0;

	for (int l = 0; l < local_L_with_ghosts; l++) {
		int l_global = l_start + l - 1; // -1 for ghost cell offset
		double z = l_global * dz;
		
		R[l] = r2(z) - r1(z);
		dr[l] = R[l] / M_max;
		
		for (int m = 0; m < M_max + 1; m++) {
			r[l][m] = (1 - m * dy) * r1(z) + m * dy * r2(z);
			r_z[l][m] = (1 - m * dy) * der_r1(z) + m * dy * der_r2(z);
		}
	}

	// initial condition
	#pragma omp parallel for collapse(2)
	for (int l = 1; l < local_L_with_ghosts - 1; l++) {
		for (int m = 0; m < M_max + 1; m++) {
			int l_global = l_start + l - 1;
			
			rho[l][m] = 1.0;
			v_z[l][m] = 0.1;
            v_r[l][m] = 0.1;
			v_phi[l][m] = 0;
			H_phi[l][m] = (1 - 0.9 * l_global * dz) * r_0 / r[l][m];
			H_z[l][m] = H_z0;
			H_r[l][m] = H_z[l][m] * r_z[l][m];
			
			e[l][m] = beta / (2.0 * (gamma - 1.0));
			p[l][m] = beta / 2.0;
			P[l][m] = p[l][m] + 1.0 / 2.0 * (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) + pow(H_phi[l][m], 2));
		}
	}

	// filling meshes for u
	#pragma omp parallel for collapse(2)
	for (int l = 1; l < local_L_with_ghosts - 1; l++) {
		for (int m = 0; m < M_max + 1; m++) {
			u0_1[l][m] = rho[l][m] * r[l][m];
			u0_2[l][m] = rho[l][m] * v_z[l][m] * r[l][m];
			u0_3[l][m] = rho[l][m] * v_r[l][m] * r[l][m];
			u0_4[l][m] = rho[l][m] * v_phi[l][m] * r[l][m];
			u0_5[l][m] = rho[l][m] * e[l][m] * r[l][m];
			u0_6[l][m] = H_phi[l][m];
			u0_7[l][m] = H_z[l][m] * r[l][m];
			u0_8[l][m] = H_r[l][m] * r[l][m];
		}
	}

	// Buffers for MPI communication
	double *send_left = new double[M_max + 1];
	double *send_right = new double[M_max + 1];
	double *recv_left = new double[M_max + 1];
	double *recv_right = new double[M_max + 1];

	// start count time
	if (rank == 0) {
		begin = MPI_Wtime();
	}

	// start time
    while (t < T) {

		// Exchange ghost cells between processes for all variables
		// We need to exchange 8 u variables
		double ***u0_arrays[8] = {&u0_1, &u0_2, &u0_3, &u0_4, &u0_5, &u0_6, &u0_7, &u0_8};
		
		for (int var = 0; var < 8; var++) {
			double **u0_var = *u0_arrays[var];
			
			for (int m = 0; m < M_max + 1; m++) {
				send_left[m] = u0_var[1][m];
				send_right[m] = u0_var[local_L][m];
			}
			
			MPI_Request req[4];
			int req_count = 0;
			
			// Send to left, receive from left
			if (rank > 0) {
				MPI_Isend(send_left, M_max + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req[req_count++]);
				MPI_Irecv(recv_left, M_max + 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &req[req_count++]);
			}
			
			// Send to right, receive from right
			if (rank < size - 1) {
				MPI_Isend(send_right, M_max + 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &req[req_count++]);
				MPI_Irecv(recv_right, M_max + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req[req_count++]);
			}
			
			MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
			
			// Copy received data to ghost cells
			if (rank > 0) {
				for (int m = 0; m < M_max + 1; m++) {
					u0_var[0][m] = recv_left[m];
				}
			}
			if (rank < size - 1) {
				for (int m = 0; m < M_max + 1; m++) {
					u0_var[local_L + 1][m] = recv_right[m];
				}
			}
		}

		// Exchange physical variables as well
		double ***phys_arrays[10] = {&rho, &v_z, &v_r, &v_phi, &e, &p, &P, &H_z, &H_r, &H_phi};
		
		for (int var = 0; var < 10; var++) {
			double **phys_var = *phys_arrays[var];
			
			for (int m = 0; m < M_max + 1; m++) {
				send_left[m] = phys_var[1][m];
				send_right[m] = phys_var[local_L][m];
			}
			
			MPI_Request req[4];
			int req_count = 0;
			
			if (rank > 0) {
				MPI_Isend(send_left, M_max + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req[req_count++]);
				MPI_Irecv(recv_left, M_max + 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &req[req_count++]);
			}
			
			if (rank < size - 1) {
				MPI_Isend(send_right, M_max + 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &req[req_count++]);
				MPI_Irecv(recv_right, M_max + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req[req_count++]);
			}
			
			MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
			
			if (rank > 0) {
				for (int m = 0; m < M_max + 1; m++) {
					phys_var[0][m] = recv_left[m];
				}
			}
			if (rank < size - 1) {
				for (int m = 0; m < M_max + 1; m++) {
					phys_var[local_L + 1][m] = recv_right[m];
				}
			}
		}

		// counting central part
		#pragma omp parallel for collapse(2)
        for (int l = 1; l < local_L + 1; l++) {
			for (int m = 1; m < M_max; m++) {
				u_1[l][m] = 0.25 * (u0_1[l + 1][m]  + u0_1[l - 1][m] + u0_1[l][m + 1] + u0_1[l][m - 1]) +
							dt * (0 - 
								  (u0_1[l + 1][m] * v_z[l + 1][m] - u0_1[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_1[l][m + 1] * v_r[l][m + 1] - u0_1[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

				u_2[l][m] = 0.25 * (u0_2[l + 1][m] + u0_2[l - 1][m] + u0_2[l][m + 1] + u0_2[l][m - 1]) +
							dt * (((pow(H_z[l + 1][m], 2) - P[l + 1][m]) * r[l + 1][m] - 
								   (pow(H_z[l - 1][m], 2) - P[l - 1][m]) * r[l - 1][m]) / (2 * dz) + 
								  ((H_z[l][m + 1] * H_r[l][m + 1]) * r[l][m + 1] - 
								   (H_z[l][m - 1] * H_r[l][m - 1]) * r[l][m - 1]) / (2 * dr[l]) - 
								  (u0_2[l + 1][m] * v_z[l + 1][m] - u0_2[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_2[l][m + 1] * v_r[l][m + 1] - u0_2[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

				u_3[l][m] = 0.25 * (u0_3[l + 1][m] + u0_3[l - 1][m] + u0_3[l][m + 1] + u0_3[l][m - 1]) +
							dt * ((rho[l][m] * pow(v_phi[l][m], 2) + P[l][m] - pow(H_phi[l][m], 2)) + 
								  (H_z[l + 1][m] * H_r[l + 1][m] * r[l + 1][m] - 
								   H_z[l - 1][m] * H_r[l - 1][m] * r[l - 1][m]) / (2 * dz) + 
								  ((pow(H_r[l][m + 1], 2) - P[l][m + 1]) * r[l][m + 1] - 
								   (pow(H_r[l][m - 1], 2) - P[l][m - 1]) * r[l][m - 1]) / (2 * dr[l]) - 
								  (u0_3[l + 1][m] * v_z[l + 1][m] - u0_3[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_3[l][m + 1] * v_r[l][m + 1] - u0_3[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

				u_4[l][m] = 0.25 * (u0_4[l + 1][m] + u0_4[l - 1][m] + u0_4[l][m + 1] + u0_4[l][m - 1]) +
							dt * ((-rho[l][m] * v_r[l][m] * v_phi[l][m] + H_phi[l][m] * H_r[l][m]) + 
								  (H_phi[l + 1][m] * H_z[l + 1][m] * r[l + 1][m] - 
								   H_phi[l - 1][m] * H_z[l - 1][m] * r[l - 1][m]) / (2 * dz) + 
								  (H_phi[l][m + 1] * H_r[l][m + 1] * r[l][m + 1] - 
								   H_phi[l][m - 1] * H_r[l][m - 1] * r[l][m - 1]) / (2 * dr[l]) - 
								  (u0_4[l + 1][m] * v_z[l + 1][m] - u0_4[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_4[l][m + 1] * v_r[l][m + 1] - u0_4[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

				u_5[l][m] = 0.25 * (u0_5[l + 1][m] + u0_5[l - 1][m] + u0_5[l][m + 1] + u0_5[l][m - 1]) +
							dt * (-p[l][m] * ((v_z[l + 1][m] * r[l + 1][m] -
											   v_z[l - 1][m] * r[l - 1][m]) / (2 * dz) + 
											  (v_r[l][m + 1] * r[l][m + 1] - 
											   v_r[l][m - 1] * r[l][m - 1]) / (2 * dr[l])) - 
								  (u0_5[l + 1][m] * v_z[l + 1][m] - u0_5[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_5[l][m + 1] * v_r[l][m + 1] - u0_5[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

				u_6[l][m] = 0.25 * (u0_6[l + 1][m] + u0_6[l - 1][m] + u0_6[l][m + 1] + u0_6[l][m - 1]) +
							dt * ((H_z[l + 1][m] * v_phi[l + 1][m] - 
								   H_z[l - 1][m] * v_phi[l - 1][m]) / (2 * dz) + 
								  (H_r[l][m + 1] * v_phi[l][m + 1] - 
								   H_r[l][m - 1] * v_phi[l][m - 1]) / (2 * dr[l]) - 
								  (u0_6[l + 1][m] * v_z[l + 1][m] - u0_6[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_6[l][m + 1] * v_r[l][m + 1] - u0_6[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

				u_7[l][m] = 0.25 * (u0_7[l + 1][m] + u0_7[l - 1][m] + u0_7[l][m + 1] + u0_7[l][m - 1]) +
							dt * ((H_r[l][m + 1] * v_z[l][m + 1] * r[l][m + 1] - 
								   H_r[l][m - 1] * v_z[l][m - 1] * r[l][m - 1]) / (2 * dr[l]) - 
								  (u0_7[l][m + 1] * v_r[l][m + 1] - u0_7[l][m - 1] * v_r[l][m - 1]) / (2 * dr[l]));

				u_8[l][m] = 0.25 * (u0_8[l + 1][m] + u0_8[l - 1][m] + u0_8[l][m + 1] + u0_8[l][m - 1]) +
							dt * ((H_z[l + 1][m] * v_r[l + 1][m] * r[l + 1][m] - 
								   H_z[l - 1][m] * v_r[l - 1][m] * r[l - 1][m]) / (2 * dz) - 
								  (u0_8[l + 1][m] * v_z[l + 1][m] - u0_8[l - 1][m] * v_z[l - 1][m]) / (2 * dz));
			}
		}

		// update central part
		#pragma omp parallel for collapse(2)
		for (int l = 1; l < local_L + 1; l++) {
			for (int m = 1; m < M_max; m++) {
				rho[l][m] = u_1[l][m] / r[l][m];
				v_z[l][m] = u_2[l][m] / u_1[l][m];
				v_r[l][m] = u_3[l][m] / u_1[l][m];
				v_phi[l][m] = u_4[l][m] / u_1[l][m];

				H_phi[l][m] = u_6[l][m];
				H_z[l][m] = u_7[l][m] / r[l][m];
				H_r[l][m] = u_8[l][m] / r[l][m];

				e[l][m] = u_5[l][m] / u_1[l][m];
				p[l][m] = (gamma - 1) * rho[l][m] * e[l][m];
				P[l][m] = p[l][m] + 1.0 / 2.0 * (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) + pow(H_phi[l][m], 2));
			}
		}

		// left boundary condition (only for rank 0)
		if (rank == 0) {
			#pragma omp parallel for
			for (int m = 0; m < M_max + 1; m++) {
				rho[1][m] = 1.0;
				v_phi[1][m] = 0;
				v_z[1][m] = u_2[2][m] / (rho[1][m] * r[1][m]);
				v_r[1][m] = 0;
				H_phi[1][m] = r_0 / r[1][m];
				H_z[1][m] = H_z0;
				H_r[1][m] = 0;
				e[1][m] = beta / (2.0 * (gamma - 1.0)) * pow(rho[1][m], gamma - 1.0);
			}

			#pragma omp parallel for
			for (int m = 0; m < M_max + 1; m++) {
				u_1[1][m] = rho[1][m] * r[1][m];
				u_2[1][m] = rho[1][m] * v_z[1][m] * r[1][m];
				u_3[1][m] = rho[1][m] * v_r[1][m] * r[1][m];
				u_4[1][m] = rho[1][m] * v_phi[1][m] * r[1][m];
				u_5[1][m] = rho[1][m] * e[1][m] * r[1][m];
				u_6[1][m] = H_phi[1][m];
				u_7[1][m] = H_z[1][m] * r[1][m];
				u_8[1][m] = H_r[1][m] * r[1][m];
			}
		}

		// up boundary condition
		#pragma omp parallel for
		for (int l = 1; l < local_L + 1; l++) {
			rho[l][M_max] = rho[l][M_max - 1];
			v_z[l][M_max] = v_z[l][M_max - 1];
			v_r[l][M_max] = v_z[l][M_max] * r_z[l][M_max];
			v_phi[l][M_max] = v_phi[l][M_max - 1];
			e[l][M_max] = e[l][M_max - 1];
			H_phi[l][M_max] = H_phi[l][M_max - 1];
			H_z[l][M_max] = H_z[l][M_max - 1];
			H_r[l][M_max] = H_z[l][M_max] * r_z[l][M_max];

			u_1[l][M_max] = rho[l][M_max] * r[l][M_max];
			u_2[l][M_max] = rho[l][M_max] * v_z[l][M_max] * r[l][M_max];
			u_3[l][M_max] = rho[l][M_max] * v_r[l][M_max] * r[l][M_max];
			u_4[l][M_max] = rho[l][M_max] * v_phi[l][M_max] * r[l][M_max];
			u_5[l][M_max] = rho[l][M_max] * e[l][M_max] * r[l][M_max];
			u_6[l][M_max] = H_phi[l][M_max];
			u_7[l][M_max] = H_z[l][M_max] * r[l][M_max];
			u_8[l][M_max] = H_r[l][M_max] * r[l][M_max];
		}

		// down boundary condition l <= L_end
		// Check if this process contains cells in the [1, L_end] range
		int local_L_end_rel = -1;
		if (l_start <= L_end && l_end >= 1) {
			int L_end_in_domain = std::min(L_end, l_end);
			local_L_end_rel = L_end_in_domain - l_start + 1;
			
			for (int l = 1; l <= local_L_end_rel; l++) {
				int l_global = l_start + l - 1;
				if (l_global >= 1 && l_global <= L_end) {
					rho[l][0] = rho[l][1];
					v_z[l][0] = v_z[l][1];
					v_r[l][0] = v_z[l][1] * r_z[l][1];
					v_phi[l][0] = v_phi[l][1];
					e[l][0] = e[l][1];
					H_phi[l][0] = H_phi[l][1];
					H_z[l][0] = H_z[l][1];
					H_r[l][0] = H_z[l][1] * r_z[l][1];

					u_1[l][0] = rho[l][0] * r[l][0];
					u_2[l][0] = rho[l][0] * v_z[l][0] * r[l][0];
					u_3[l][0] = rho[l][0] * v_r[l][0] * r[l][0];
					u_4[l][0] = rho[l][0] * v_phi[l][0] * r[l][0];
					u_5[l][0] = rho[l][0] * e[l][0] * r[l][0];
					u_6[l][0] = H_phi[l][0];
					u_7[l][0] = H_z[l][0] * r[l][0];
					u_8[l][0] = H_r[l][0] * r[l][0];
				}
			}
		}

		// down boundary condition l > L_end
		#pragma omp parallel for
		for (int l = 1; l < local_L + 1; l++) {
			int l_global = l_start + l - 1;
			
			if (l_global > L_end && l_global < L_max_global) {
				int m = 0;
				
				u_1[l][m] = (0.25 * (u0_1[l + 1][m] / r[l + 1][m] + u0_1[l - 1][m] / r[l - 1][m] + u0_1[l][m + 1] / r[l][m + 1] + u0_1[l][m] / r[l][m]) +
							dt * (0 - 
								  (u0_1[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] - u0_1[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_1[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] - u0_1[l][m] / r[l][m] * (v_r[l][1])) / (dr[l]))
							) * r[l][m];

				u_2[l][m] = (0.25 * (u0_2[l + 1][m] / r[l + 1][m] + u0_2[l - 1][m] / r[l - 1][m] + u0_2[l][m + 1] / r[l][m + 1] + u0_2[l][m] / r[l][m]) +
							dt * (((pow(H_z[l + 1][m], 2) - P[l + 1][m]) - 
								   (pow(H_z[l - 1][m], 2) - P[l - 1][m])) / (2 * dz) + 
								  ((H_z[l][m + 1] * H_r[l][m + 1]) - 
								   (H_z[l][m] * (H_r[l][m]))) / (dr[l]) - 
								  (u0_2[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] - u0_2[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_2[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] - u0_2[l][m] / r[l][m] * (v_r[l][m])) / (dr[l]))
							) * r[l][m];

				u_3[l][m] = 0;
				u_4[l][m] = 0;

				u_5[l][m] = (0.25 * (u0_5[l + 1][m] / r[l + 1][m] + u0_5[l - 1][m] / r[l - 1][m] + u0_5[l][m + 1] / r[l][m + 1] + u0_5[l][m] / r[l][m]) +
							dt * (-p[l][m] * ((v_z[l + 1][m] -
											   v_z[l - 1][m]) / (2 * dz) + 
											  (v_r[l][m + 1] - 
											   (v_r[l][m])) / (dr[l])) - 
								  (u0_5[l + 1][m] / r[l + 1][m] * v_z[l + 1][m] - u0_5[l - 1][m] / r[l - 1][m] * v_z[l - 1][m]) / (2 * dz) - 
								  (u0_5[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] - u0_5[l][m] / r[l][m] * (v_r[l][m])) / (dr[l]))
							) * r[l][m];

				u_6[l][m] = 0;

				u_7[l][m] = (0.25 * (u0_7[l + 1][m] / r[l + 1][m] + u0_7[l - 1][m] / r[l - 1][m] + u0_7[l][m + 1] / r[l][m + 1] + u0_7[l][m] / r[l][m]) +
							dt * ((H_r[l][m + 1] * v_z[l][m + 1] - 
								   (H_r[l][m]) * v_z[l][m]) / (dr[l]) - 
								  (u0_7[l][m + 1] / r[l][m + 1] * v_r[l][m + 1] - u0_7[l][m] / r[l][m] * (v_r[l][m])) / (dr[l]))
							) * r[l][m];

				u_8[l][m] = 0;
			}
		}

		// right boundary condition (only for last rank)
		if (rank == size - 1) {
			#pragma omp parallel for
			for (int m = 0; m < M_max + 1; m++) {
				u_1[local_L][m] = u_1[local_L - 1][m];
				u_2[local_L][m] = u_2[local_L - 1][m];
				u_3[local_L][m] = u_3[local_L - 1][m];
				u_4[local_L][m] = u_4[local_L - 1][m];
				u_5[local_L][m] = u_5[local_L - 1][m];
				u_6[local_L][m] = u_6[local_L - 1][m];
				u_7[local_L][m] = u_7[local_L - 1][m];
				u_8[local_L][m] = u_8[local_L - 1][m];
			}
		}

        // data update
		#pragma omp parallel for collapse(2)
		for (int l = 1; l < local_L + 1; l++) {
			for (int m = 0; m < M_max + 1; m++) {
				rho[l][m] = u_1[l][m] / r[l][m];
				v_z[l][m] = u_2[l][m] / u_1[l][m];
				v_r[l][m] = u_3[l][m] / u_1[l][m];
				v_phi[l][m] = u_4[l][m] / u_1[l][m];

				H_phi[l][m] = u_6[l][m];
				H_z[l][m] = u_7[l][m] / r[l][m];
				H_r[l][m] = u_8[l][m] / r[l][m];

				e[l][m] = u_5[l][m] / u_1[l][m];
				p[l][m] = (gamma - 1) * rho[l][m] * e[l][m];
				P[l][m] = p[l][m] + 1.0 / 2.0 * (pow(H_z[l][m], 2) + pow(H_r[l][m], 2) + pow(H_phi[l][m], 2));
			}
		}

		#pragma omp parallel for collapse(2)
		for (int l = 0; l < local_L_with_ghosts; l++) {
			for (int m = 0; m < M_max + 1; m++) {
				u0_1[l][m] = u_1[l][m];
				u0_2[l][m] = u_2[l][m];
				u0_3[l][m] = u_3[l][m];
				u0_4[l][m] = u_4[l][m];
				u0_5[l][m] = u_5[l][m];
				u0_6[l][m] = u_6[l][m];
				u0_7[l][m] = u_7[l][m];
				u0_8[l][m] = u_8[l][m];
			}
		}

		// animation output
		if ((int)(t * 10000) % 1000 == 0 && animate == 1) {
			animate_write((int)(t * 10000), local_L, M_max, dz, r, rho, v_z, v_r, v_phi, e, H_z, H_r, H_phi, rank, size, l_start);
		}

        // time step
        t += dt;

		// checkout
		if (rank == 0) {
			int check_l = 20;
			int check_m = 40;
			if (check_l >= l_start && check_l <= l_end) {
				int local_check_l = check_l - l_start + 1;
				printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", t, rho[local_check_l][check_m], 
					v_z[local_check_l][check_m], v_phi[local_check_l][check_m], 
					e[local_check_l][check_m], H_phi[local_check_l][check_m]);
			}
		}
    }

	// finish count time
	if (rank == 0) {
		end = MPI_Wtime();
		total = end - begin;
		printf("Calculation time : %lf sec\n", total);
	}

	// Gather results to rank 0 for final output
	// Allocate global arrays on rank 0
	double **rho_global = nullptr, **v_z_global = nullptr, **v_r_global = nullptr;
	double **v_phi_global = nullptr, **e_global = nullptr;
	double **H_z_global = nullptr, **H_r_global = nullptr, **H_phi_global = nullptr;
	double **r_global = nullptr;
	
	if (rank == 0) {
		memory_allocation_2D(rho_global, L_max_global + 1, M_max + 1);
		memory_allocation_2D(v_z_global, L_max_global + 1, M_max + 1);
		memory_allocation_2D(v_r_global, L_max_global + 1, M_max + 1);
		memory_allocation_2D(v_phi_global, L_max_global + 1, M_max + 1);
		memory_allocation_2D(e_global, L_max_global + 1, M_max + 1);
		memory_allocation_2D(H_z_global, L_max_global + 1, M_max + 1);
		memory_allocation_2D(H_r_global, L_max_global + 1, M_max + 1);
		memory_allocation_2D(H_phi_global, L_max_global + 1, M_max + 1);
		memory_allocation_2D(r_global, L_max_global + 1, M_max + 1);
	}

	// Gather data row by row
	for (int m = 0; m < M_max + 1; m++) {
		double *local_row_rho = new double[local_L];
		double *local_row_vz = new double[local_L];
		double *local_row_vr = new double[local_L];
		double *local_row_vphi = new double[local_L];
		double *local_row_e = new double[local_L];
		double *local_row_Hz = new double[local_L];
		double *local_row_Hr = new double[local_L];
		double *local_row_Hphi = new double[local_L];
		double *local_row_r = new double[local_L];
		
		for (int l = 0; l < local_L; l++) {
			local_row_rho[l] = rho[l + 1][m];
			local_row_vz[l] = v_z[l + 1][m];
			local_row_vr[l] = v_r[l + 1][m];
			local_row_vphi[l] = v_phi[l + 1][m];
			local_row_e[l] = e[l + 1][m];
			local_row_Hz[l] = H_z[l + 1][m];
			local_row_Hr[l] = H_r[l + 1][m];
			local_row_Hphi[l] = H_phi[l + 1][m];
			local_row_r[l] = r[l + 1][m];
		}
		
		double *global_row_rho = nullptr, *global_row_vz = nullptr, *global_row_vr = nullptr;
		double *global_row_vphi = nullptr, *global_row_e = nullptr;
		double *global_row_Hz = nullptr, *global_row_Hr = nullptr, *global_row_Hphi = nullptr;
		double *global_row_r = nullptr;
		
		if (rank == 0) {
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
		
		int *recvcounts = new int[size];
		int *displs = new int[size];
		
		for (int i = 0; i < size; i++) {
			int i_start = i * L_per_proc;
			int i_end = (i + 1) * L_per_proc - 1;
			if (i == size - 1) i_end = L_max_global - 1;
			recvcounts[i] = i_end - i_start + 1;
			displs[i] = i_start;
		}
		
		MPI_Gatherv(local_row_rho, local_L, MPI_DOUBLE, global_row_rho, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_row_vz, local_L, MPI_DOUBLE, global_row_vz, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_row_vr, local_L, MPI_DOUBLE, global_row_vr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_row_vphi, local_L, MPI_DOUBLE, global_row_vphi, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_row_e, local_L, MPI_DOUBLE, global_row_e, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_row_Hz, local_L, MPI_DOUBLE, global_row_Hz, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_row_Hr, local_L, MPI_DOUBLE, global_row_Hr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_row_Hphi, local_L, MPI_DOUBLE, global_row_Hphi, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(local_row_r, local_L, MPI_DOUBLE, global_row_r, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		if (rank == 0) {
			for (int l = 0; l < L_max_global; l++) {
				rho_global[l][m] = global_row_rho[l];
				v_z_global[l][m] = global_row_vz[l];
				v_r_global[l][m] = global_row_vr[l];
				v_phi_global[l][m] = global_row_vphi[l];
				e_global[l][m] = global_row_e[l];
				H_z_global[l][m] = global_row_Hz[l];
				H_r_global[l][m] = global_row_Hr[l];
				H_phi_global[l][m] = global_row_Hphi[l];
				r_global[l][m] = global_row_r[l];
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

	// output results in file (only rank 0)
	if (rank == 0) {
		std :: ofstream out("28-2D_MHD_LF_rzphi_MPD_MPI.plt");
		int np = (L_max_global + 1) * (M_max + 1);
		int ne = L_max_global * M_max;
		double hfr;
		out << "VARIABLES=\n";
		out << "\"z\"\n\"r\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n\"Energy\"\n\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
		out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne << "\n ";
		for (int m = 0; m < M_max + 1; m++) {
			for (int l = 0; l < L_max_global + 1; l++) {
				if (l < L_max_global) {
					hfr = H_phi_global[l][m] * r_global[l][m];
					out << l * dz << " " << r_global[l][m] << " " << rho_global[l][m] << " " << 
						v_z_global[l][m] << " " << v_r_global[l][m] << " " << std::sqrt(v_z_global[l][m] * v_z_global[l][m] + v_r_global[l][m] * v_r_global[l][m]) << " " << 
						v_phi_global[l][m] << " " << e_global[l][m] << " " << H_z_global[l][m] << " " << H_r_global[l][m] << " " << hfr << " " << H_phi_global[l][m] << "\n";
				} else {
					// Last row - copy from L_max_global-1
					hfr = H_phi_global[L_max_global-1][m] * r_global[L_max_global-1][m];
					out << l * dz << " " << r_global[L_max_global-1][m] << " " << rho_global[L_max_global-1][m] << " " << 
						v_z_global[L_max_global-1][m] << " " << v_r_global[L_max_global-1][m] << " " << std::sqrt(v_z_global[L_max_global-1][m] * v_z_global[L_max_global-1][m] + v_r_global[L_max_global-1][m] * v_r_global[L_max_global-1][m]) << " " << 
						v_phi_global[L_max_global-1][m] << " " << e_global[L_max_global-1][m] << " " << H_z_global[L_max_global-1][m] << " " << H_r_global[L_max_global-1][m] << " " << hfr << " " << H_phi_global[L_max_global-1][m] << "\n";
				}
			}
		}

		int i1 = 0;
		int i2 = 0;
		int i3 = 0;
		int i4 = 0;
		
		for (int m = 0; m < M_max; m++) {
			for (int l = 0; l < L_max_global; l++) {
				i1 = l + m * (L_max_global + 1) + 1;
				i2 = l + 1 + m * (L_max_global + 1) + 1;
				i3 = l + 1 + (m + 1) * (L_max_global + 1) + 1;
				i4 = l + (m + 1) * (L_max_global + 1) + 1;
				out << i1 << " " << i2 << " " << i3 << " " << i4 << "\n";
			}
		}

		out.close();

		// clear global memory
		memory_clearing_2D(rho_global, L_max_global + 1);
		memory_clearing_2D(v_z_global, L_max_global + 1);
		memory_clearing_2D(v_r_global, L_max_global + 1);
		memory_clearing_2D(v_phi_global, L_max_global + 1);
		memory_clearing_2D(e_global, L_max_global + 1);
		memory_clearing_2D(H_z_global, L_max_global + 1);
		memory_clearing_2D(H_r_global, L_max_global + 1);
		memory_clearing_2D(H_phi_global, L_max_global + 1);
		memory_clearing_2D(r_global, L_max_global + 1);
	}

	// clear local memory
    memory_clearing_2D(u_1, local_L_with_ghosts);
	memory_clearing_2D(u_2, local_L_with_ghosts);
    memory_clearing_2D(u_3, local_L_with_ghosts);
	memory_clearing_2D(u_4, local_L_with_ghosts);
	memory_clearing_2D(u_5, local_L_with_ghosts);
	memory_clearing_2D(u_6, local_L_with_ghosts);
	memory_clearing_2D(u_7, local_L_with_ghosts);
	memory_clearing_2D(u_8, local_L_with_ghosts);

	memory_clearing_2D(u0_1, local_L_with_ghosts);
	memory_clearing_2D(u0_2, local_L_with_ghosts);
	memory_clearing_2D(u0_3, local_L_with_ghosts);
	memory_clearing_2D(u0_4, local_L_with_ghosts);
	memory_clearing_2D(u0_5, local_L_with_ghosts);
	memory_clearing_2D(u0_6, local_L_with_ghosts);
	memory_clearing_2D(u0_7, local_L_with_ghosts);
	memory_clearing_2D(u0_8, local_L_with_ghosts);

	memory_clearing_2D(rho, local_L_with_ghosts);
	memory_clearing_2D(v_r, local_L_with_ghosts);
	memory_clearing_2D(v_phi, local_L_with_ghosts);
	memory_clearing_2D(v_z, local_L_with_ghosts);
	memory_clearing_2D(e, local_L_with_ghosts);
	memory_clearing_2D(p, local_L_with_ghosts);
	memory_clearing_2D(P, local_L_with_ghosts);
	memory_clearing_2D(H_r, local_L_with_ghosts);
	memory_clearing_2D(H_phi, local_L_with_ghosts);
	memory_clearing_2D(H_z, local_L_with_ghosts);

	memory_clearing_2D(r, local_L_with_ghosts);
	memory_clearing_2D(r_z, local_L_with_ghosts);

	delete[] R;
	delete[] dr;
	delete[] send_left;
	delete[] send_right;
	delete[] recv_left;
	delete[] recv_right;

	MPI_Finalize();
	return 0;
}
