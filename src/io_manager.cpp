#include "io_manager.hpp"

#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

// ============================================================
// Constructor
// ============================================================

IOManager::IOManager(const SimConfig& cfg, const MPIManager& mpi)
    : cfg_(cfg), mpi_(mpi)
{}

// ============================================================
// Per-rank animation frame (Tecplot)
// ============================================================

void IOManager::write_animate_frame(int step, const Fields& f,
                                     const Grid& grid) const {
    char filename[256];
    std::snprintf(filename, sizeof(filename),
                  "animate_m_29_800x400/%d_rank%d.plt", step, mpi_.rank);

    std::ofstream out(filename);
    if (!out) return;

    const int local_L = mpi_.local_L;
    const int M_max   = cfg_.M_max;
    const int np      = (local_L + 1) * (M_max + 1);
    const int ne      = local_L * M_max;

    out << "VARIABLES=\n"
           "\"r\"\n\"z\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n"
           "\"Energy\"\n\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
    out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne << "\n ";

    for (int m = 0; m < M_max + 1; ++m) {
        for (int l = 0; l < local_L + 1; ++l) {
            const double hfr = f.H_phi[l][m] * grid.r[l][m];
            const double vl  = std::sqrt(f.v_z[l][m]*f.v_z[l][m]
                                       + f.v_r[l][m]*f.v_r[l][m]);
            out << (mpi_.l_start + l) * cfg_.dz << " "
                << grid.r[l][m]   << " " << f.rho [l][m] << " "
                << f.v_z [l][m]   << " " << f.v_r  [l][m] << " "
                << vl             << " " << f.v_phi[l][m] << " "
                << f.e   [l][m]   << " " << f.H_z  [l][m] << " "
                << f.H_r [l][m]   << " " << hfr            << " "
                << f.H_phi[l][m]  << "\n";
        }
    }

    for (int m = 0; m < M_max; ++m) {
        for (int l = 0; l < local_L; ++l) {
            const int i1 = l     +  m    * (local_L + 1) + 1;
            const int i2 = l + 1 +  m    * (local_L + 1) + 1;
            const int i3 = l + 1 + (m+1) * (local_L + 1) + 1;
            const int i4 = l     + (m+1) * (local_L + 1) + 1;
            out << i1 << " " << i2 << " " << i3 << " " << i4 << "\n";
        }
    }
}

// ============================================================
// Gather local → global
// ============================================================

void IOManager::gather_global(const Fields& f, const Grid& grid) {
    const int L_g   = cfg_.L_max_global;
    const int M_max = cfg_.M_max;

    if (mpi_.rank == 0) {
        rho_g_ .resize(L_g + 1, M_max + 1);
        v_z_g_ .resize(L_g + 1, M_max + 1);
        v_r_g_ .resize(L_g + 1, M_max + 1);
        v_phi_g_.resize(L_g + 1, M_max + 1);
        e_g_   .resize(L_g + 1, M_max + 1);
        H_z_g_ .resize(L_g + 1, M_max + 1);
        H_r_g_ .resize(L_g + 1, M_max + 1);
        H_phi_g_.resize(L_g + 1, M_max + 1);
        r_g_   .resize(L_g + 1, M_max + 1);
    }

    const int local_L  = mpi_.local_L;

    // Build recvcounts / displs once
    std::vector<int> recvcounts(mpi_.size), displs(mpi_.size);
    for (int i = 0; i < mpi_.size; ++i) {
        const int is  = i * mpi_.L_per_proc;
        int       ie  = (i + 1) * mpi_.L_per_proc - 1;
        if (i == mpi_.size - 1) ie = L_g - 1;
        recvcounts[i] = ie - is + 1;
        displs[i]     = is;
    }

    // Scratch rows
    std::vector<double> local_row(local_L);
    std::vector<double> global_row(mpi_.rank == 0 ? L_g : 0);

    // Helper: gather one 2-D field column-by-column
    auto gather_field = [&](const Array2D& src, Array2D& dst) {
        for (int m = 0; m < M_max + 1; ++m) {
            for (int l = 0; l < local_L; ++l)
                local_row[l] = src[l + 1][m];   // skip left ghost

            MPI_Gatherv(local_row.data(),  local_L, MPI_DOUBLE,
                        global_row.data(), recvcounts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (mpi_.rank == 0)
                for (int l = 0; l < L_g; ++l)
                    dst[l][m] = global_row[l];
        }
    };

    gather_field(f.rho,   rho_g_);
    gather_field(f.v_z,   v_z_g_);
    gather_field(f.v_r,   v_r_g_);
    gather_field(f.v_phi, v_phi_g_);
    gather_field(f.e,     e_g_);
    gather_field(f.H_z,   H_z_g_);
    gather_field(f.H_r,   H_r_g_);
    gather_field(f.H_phi, H_phi_g_);
    gather_field(grid.r,  r_g_);
}

// ============================================================
// VTK output (rank 0 only)
// ============================================================

void IOManager::write_vtk(const std::string& filename) const {
    if (mpi_.rank != 0) return;

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening " << filename << "\n";
        return;
    }

    const int L_g   = cfg_.L_max_global;
    const int M_max = cfg_.M_max;
    const int nx    = L_g + 1;
    const int ny    = M_max + 1;

    out << "# vtk DataFile Version 2.0\n"
           "MHD Simulation\nASCII\nDATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << nx << " " << ny << " 1\n";
    out << "POINTS " << nx * ny << " float\n";

    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            const int li = std::min(l, L_g - 1);
            out << r_g_[li][m] << " " << l * cfg_.dz << " 0.0\n";
        }
    }

    out << "POINT_DATA " << nx * ny << "\n";

    auto write_scalar = [&](const char* name, const Array2D& arr) {
        out << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
        for (int m = 0; m < ny; ++m)
            for (int l = 0; l < nx; ++l)
                out << arr[std::min(l, L_g-1)][m] << "\n";
    };

    auto write_vector = [&](const char* name, const Array2D& ax,
                             const Array2D& ay) {
        out << "VECTORS " << name << " float\n";
        for (int m = 0; m < ny; ++m)
            for (int l = 0; l < nx; ++l) {
                const int li = std::min(l, L_g - 1);
                out << ax[li][m] << " " << ay[li][m] << " 0.0\n";
            }
    };

    write_scalar("Rho",    rho_g_);
    write_vector("Velocity", v_r_g_, v_z_g_);
    write_scalar("Vphi",   v_phi_g_);
    write_scalar("Energy", e_g_);
    write_vector("MagneticField", H_r_g_, H_z_g_);
    write_scalar("Hphi",   H_phi_g_);

    // Derived: |v|
    out << "SCALARS Vl float 1\nLOOKUP_TABLE default\n";
    for (int m = 0; m < ny; ++m)
        for (int l = 0; l < nx; ++l) {
            const int li = std::min(l, L_g - 1);
            out << std::sqrt(v_z_g_[li][m]*v_z_g_[li][m]
                           + v_r_g_[li][m]*v_r_g_[li][m]) << "\n";
        }

    // Derived: H_phi * r
    out << "SCALARS Hphi_r float 1\nLOOKUP_TABLE default\n";
    for (int m = 0; m < ny; ++m)
        for (int l = 0; l < nx; ++l) {
            const int li = std::min(l, L_g - 1);
            out << H_phi_g_[li][m] * r_g_[li][m] << "\n";
        }
}

// ============================================================
// Tecplot FEPOINT output (rank 0 only)
// ============================================================

void IOManager::write_tecplot(const std::string& filename) const {
    if (mpi_.rank != 0) return;

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening " << filename << "\n";
        return;
    }

    const int L_g   = cfg_.L_max_global;
    const int M_max = cfg_.M_max;
    const int np    = (L_g + 1) * (M_max + 1);
    const int ne    =  L_g      *  M_max;

    out << "VARIABLES=\n"
           "\"z\"\n\"r\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n"
           "\"Energy\"\n\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
    out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne << "\n ";

    for (int m = 0; m < M_max + 1; ++m) {
        for (int l = 0; l < L_g + 1; ++l) {
            const int li  = std::min(l, L_g - 1);
            const double vl  = std::sqrt(v_z_g_[li][m]*v_z_g_[li][m]
                                       + v_r_g_[li][m]*v_r_g_[li][m]);
            const double hfr = H_phi_g_[li][m] * r_g_[li][m];
            out << l * cfg_.dz    << " " << r_g_  [li][m] << " "
                << rho_g_ [li][m] << " " << v_z_g_[li][m] << " "
                << v_r_g_ [li][m] << " " << vl             << " "
                << v_phi_g_[li][m]<< " " << e_g_  [li][m] << " "
                << H_z_g_ [li][m] << " " << H_r_g_[li][m] << " "
                << hfr            << " " << H_phi_g_[li][m]<< "\n";
        }
    }

    for (int m = 0; m < M_max; ++m) {
        for (int l = 0; l < L_g; ++l) {
            const int i1 = l     +  m    * (L_g + 1) + 1;
            const int i2 = l + 1 +  m    * (L_g + 1) + 1;
            const int i3 = l + 1 + (m+1) * (L_g + 1) + 1;
            const int i4 = l     + (m+1) * (L_g + 1) + 1;
            out << i1 << " " << i2 << " " << i3 << " " << i4 << "\n";
        }
    }
}
