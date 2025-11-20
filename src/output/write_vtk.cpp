#include "output/write_vtk.hpp"

#include <cmath>
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

    // Write points: r as x, z as y, 0 as z_coord
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double x = (l < L_max_global) ? r_global[l][m] : r_global[L_max_global-1][m];
            double y = l * dz;
            double z_coord = 0.0;
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

    // VECTORS Velocity (v_r, v_z, 0)
    out << "VECTORS Velocity float" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double vr = (l < L_max_global) ? v_r_global[l][m] : v_r_global[L_max_global-1][m];
            double vz = (l < L_max_global) ? v_z_global[l][m] : v_z_global[L_max_global-1][m];
            out << vr << " " << vz << " 0.0" << '\n';
        }
    }

    // SCALARS Vphi
    out << "SCALARS Vphi float 1" << '\n';
    out << "LOOKUP_TABLE default" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double val = (l < L_max_global) ? v_phi_global[l][m] : v_phi_global[L_max_global-1][m];
            out << val << '\n';
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

    // VECTORS MagneticField (H_r, H_z, 0)
    out << "VECTORS MagneticField float" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double hr = (l < L_max_global) ? H_r_global[l][m] : H_r_global[L_max_global-1][m];
            double hz = (l < L_max_global) ? H_z_global[l][m] : H_z_global[L_max_global-1][m];
            out << hr << " " << hz << " 0.0" << '\n';
        }
    }

    // SCALARS Hphi
    out << "SCALARS Hphi float 1" << '\n';
    out << "LOOKUP_TABLE default" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double val = (l < L_max_global) ? H_phi_global[l][m] : H_phi_global[L_max_global-1][m];
            out << val << '\n';
        }
    }

    // SCALARS Vl = sqrt(v_z^2 + v_r^2)
    out << "SCALARS Vl float 1" << '\n';
    out << "LOOKUP_TABLE default" << '\n';
    for (int m = 0; m < ny; ++m) {
        for (int l = 0; l < nx; ++l) {
            double vr = (l < L_max_global) ? v_r_global[l][m] : v_r_global[L_max_global-1][m];
            double vz = (l < L_max_global) ? v_z_global[l][m] : v_z_global[L_max_global-1][m];
            double vl = std::sqrt(vz * vz + vr * vr);
            out << vl << '\n';
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
