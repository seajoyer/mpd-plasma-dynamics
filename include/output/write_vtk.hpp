#pragma once

#include <string>

void WriteVtk(const char *filename, int L_max_global, int M_max, double dz,
               double **r_global, double **rho_global, double **v_z_global, 
               double **v_r_global, double **v_phi_global, double **e_global, 
               double **H_z_global, double **H_r_global, double **H_phi_global,
               const std::string& output_dir = ".");
