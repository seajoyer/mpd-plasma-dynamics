#include "grid.hpp"

#include <cmath>

Grid::Grid(const SimConfig& cfg_, int lwg, int ls)
    : cfg(cfg_),
      local_L_with_ghosts(lwg),
      l_start(ls),
      r  (lwg, cfg_.M_max + 1),
      r_z(lwg, cfg_.M_max + 1),
      R  (lwg, 0.0),
      dr (lwg, 0.0)
{
    r_0 = (r1(0.0) + r2(0.0)) / 2.0;
    build();
}

void Grid::build() {
    const double dy = cfg.dy;
    const double dz = cfg.dz;

    for (int l = 0; l < local_L_with_ghosts; ++l) {
        // l == 0 is the left ghost cell; its global index is l_start - 1.
        const int    l_global = l_start + l - 1;
        const double z        = l_global * dz;

        R[l]  = r2(z) - r1(z);
        dr[l] = R[l] / cfg.M_max;

        for (int m = 0; m < cfg.M_max + 1; ++m) {
            r  [l][m] = (1.0 - m * dy) * r1(z) + m * dy * r2(z);
            r_z[l][m] = (1.0 - m * dy) * der_r1(z) + m * dy * der_r2(z);
        }
    }
}

// ---- geometry functions -------------------------------------------------------

double Grid::r1(double z) {
    if      (z < 0.3)   return 0.2;
    else if (z < 0.4)   return 0.2 - 10.0 * std::pow(z - 0.3, 2);
    else if (z < 0.478) return 10.0 * std::pow(z - 0.5, 2);
    else                return 0.005;
}

double Grid::r2(double /*z*/) { return 0.8; }

double Grid::der_r1(double z) {
    if      (z < 0.3)   return  0.0;
    else if (z < 0.4)   return -20.0 * (z - 0.3);
    else if (z < 0.478) return  20.0 * (z - 0.5);
    else                return  0.0;
}

double Grid::der_r2(double /*z*/) { return 0.0; }
