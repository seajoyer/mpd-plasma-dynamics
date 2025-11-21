#include "output/write_plt.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

void WritePlt(const char* filename, int L_max_global, int M_max, double dz,
               double** r_global, double** rho_global, double** v_z_global,
               double** v_r_global, double** v_phi_global, double** e_global,
               double** H_z_global, double** H_r_global, double** H_phi_global,
               const std::string& output_dir) {
    
    // Create output directory if it doesn't exist
    if (output_dir != ".") {
        struct stat st = {0};
        if (stat(output_dir.c_str(), &st) == -1) {
            mkdir(output_dir.c_str(), 0755);
        }
    }
    
    // Construct full path
    std::string full_path = output_dir + "/" + filename;
    
    std::ofstream out(full_path);
    if (!out) {
        std::cerr << "Error opening " << full_path << '\n';
        return;
    }

    int np = (L_max_global + 1) * (M_max + 1);
    int ne = L_max_global * M_max;
    double hfr;

    out << "VARIABLES=\n";
    out << "\"z\"\n\"r\"\n\"Rho\"\n\"Vz\"\n\"Vr\"\n\"Vl\"\n\"Vphi\"\n\"Energy\"\n"
           "\"Hz\"\n\"Hr\"\n\"Hphi*r\"\n\"Hphi\"\n";
    out << "ZONE \n F=FEPOINT, ET=Quadrilateral, N=" << np << " E=" << ne << "\n ";

    for (int m = 0; m < M_max + 1; m++) {
        for (int l = 0; l < L_max_global + 1; l++) {
            if (l < L_max_global) {
                hfr = H_phi_global[l][m] * r_global[l][m];
                out << l * dz << " " << r_global[l][m] << " " << rho_global[l][m] << " "
                    << v_z_global[l][m] << " " << v_r_global[l][m] << " "
                    << std::sqrt(v_z_global[l][m] * v_z_global[l][m] +
                                 v_r_global[l][m] * v_r_global[l][m])
                    << " " << v_phi_global[l][m] << " " << e_global[l][m] << " "
                    << H_z_global[l][m] << " " << H_r_global[l][m] << " " << hfr << " "
                    << H_phi_global[l][m] << "\n";
            } else {
                hfr = H_phi_global[L_max_global - 1][m] * r_global[L_max_global - 1][m];
                out << l * dz << " " << r_global[L_max_global - 1][m] << " "
                    << rho_global[L_max_global - 1][m] << " "
                    << v_z_global[L_max_global - 1][m] << " "
                    << v_r_global[L_max_global - 1][m] << " "
                    << std::sqrt(v_z_global[L_max_global - 1][m] *
                                     v_z_global[L_max_global - 1][m] +
                                 v_r_global[L_max_global - 1][m] *
                                     v_r_global[L_max_global - 1][m])
                    << " " << v_phi_global[L_max_global - 1][m] << " "
                    << e_global[L_max_global - 1][m] << " "
                    << H_z_global[L_max_global - 1][m] << " "
                    << H_r_global[L_max_global - 1][m] << " " << hfr << " "
                    << H_phi_global[L_max_global - 1][m] << "\n";
            }
        }
    }

    int i1, i2, i3, i4;
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
}
