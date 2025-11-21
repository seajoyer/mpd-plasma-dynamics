#include "output/write_vtk.hpp"

#include <fstream>
#include <iostream>

void WriteVtk(const char* filename, int L_max_global, int M_max, double dz,
               double** r_global, double** rho_global, double** v_z_global,
               double** v_r_global, double** v_phi_global, double** e_global,
               double** H_z_global, double** H_r_global, double** H_phi_global) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening " << filename << '\n';
        return;
    }

    out << "# vtk DataFile Version 2.0" << '\n';
    out << "MHD Simulation" << '\n';
    out << "ASCII" << '\n';
    out << "DATASET STRUCTURED_GRID" << '\n';

    int nx = L_max_global + 1; // z direction
    int ny = M_max + 1;        // r direction
    int nz = 1;
    out << "DIMENSIONS " << nx << " " << ny << " " << nz << '\n';

    int npoints = nx * ny * nz;
    out << "POINTS " << npoints << " float" << '\n';

    // Write points: 0 as x, r as y, z as z_coord
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double x = 0.0;
            double y = (l < L_max_global) ? r_global[l][m] : r_global[L_max_global-1][m];
            double z_coord = l * dz;
            out << x << " " << y << " " << z_coord << '\n';
        }
    }

    out << "POINT_DATA " << npoints << '\n';

    // SCALARS Rho
    out << "SCALARS Rho float 1" << '\n';
    out << "LOOKUP_TABLE default" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double val = (l < L_max_global) ? rho_global[l][m] : rho_global[L_max_global-1][m];
            out << val << '\n';
        }
    }

    // VECTORS Velocity (v_phi, v_r, v_z)
    out << "VECTORS Velocity float" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double vr = (l < L_max_global) ? v_r_global[l][m] : v_r_global[L_max_global-1][m];
            double vz = (l < L_max_global) ? v_z_global[l][m] : v_z_global[L_max_global-1][m];
            double v_phi = (l < L_max_global) ? v_phi_global[l][m] : v_phi_global[L_max_global-1][m];
            out << v_phi << " " << vr << " " << vz << '\n';
        }
    }

    // SCALARS Energy
    out << "SCALARS Energy float 1" << '\n';
    out << "LOOKUP_TABLE default" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double val = (l < L_max_global) ? e_global[l][m] : e_global[L_max_global-1][m];
            out << val << '\n';
        }
    }

    // VECTORS MagneticField (H_phi, H_r, H_z)
    out << "VECTORS MagneticField float" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double hr = (l < L_max_global) ? H_r_global[l][m] : H_r_global[L_max_global-1][m];
            double hz = (l < L_max_global) ? H_z_global[l][m] : H_z_global[L_max_global-1][m];
            double h_phi = (l < L_max_global) ? H_phi_global[l][m] : H_phi_global[L_max_global-1][m];
            out << h_phi << " " << hr << " " << hz << '\n';
        }
    }

    // SCALARS Hphi*r
    out << "SCALARS Hphi_r float 1" << '\n';
    out << "LOOKUP_TABLE default" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double hphi = (l < L_max_global) ? H_phi_global[l][m] : H_phi_global[L_max_global-1][m];
            double r_val = (l < L_max_global) ? r_global[l][m] : r_global[L_max_global-1][m];
            out << hphi * r_val << '\n';
        }
    }

    out.close();
}
