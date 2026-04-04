#include "grid.hpp"

#include <cmath>

Grid::Grid(const SimConfig& cfg_, int lwg, int ls, int lmwg, int ms)
    : cfg(cfg_),
      local_L_with_ghosts(lwg),
      l_start(ls),
      local_M_with_ghosts(lmwg),
      m_start(ms),
      r  (lwg, lmwg),
      r_z(lwg, lmwg),
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
        dr[l] = R[l] / cfg.M_max;   // global radial cell spacing

        for (int m = 0; m < local_M_with_ghosts; ++m) {
            // m == 0 is the inner ghost column; its global index is m_start - 1.
            const int    m_global = m_start + m - 1;
            const double frac     = m_global * dy;   // == m_global / M_max
            r  [l][m] = (1.0 - frac) * r1(z) + frac * r2(z);
            r_z[l][m] = (1.0 - frac) * der_r1(z) + frac * der_r2(z);
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
