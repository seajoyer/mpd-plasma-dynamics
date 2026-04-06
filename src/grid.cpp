#include "grid.hpp"

Grid::Grid(const SimConfig& cfg_, int lwg, int ls, int lmwg, int ms,
           const IGeometry& geom_)
    : cfg(cfg_),
      geom(geom_),
      local_L_with_ghosts(lwg),
      l_start(ls),
      local_M_with_ghosts(lmwg),
      m_start(ms),
      r  (lwg, lmwg),
      r_z(lwg, lmwg),
      R  (lwg, 0.0),
      dr (lwg, 0.0)
{
    r_0 = (geom.r_inner(0.0) + geom.r_outer(0.0)) / 2.0;
    build();
}

void Grid::build() {
    const double dy = cfg.dy;
    const double dz = cfg.dz;

    for (int l = 0; l < local_L_with_ghosts; ++l) {
        // l == 0 is the left ghost cell; its global index is l_start - 1.
        const int    l_global = l_start + l - 1;
        const double z        = l_global * dz;

        R[l]  = geom.r_outer(z) - geom.r_inner(z);
        dr[l] = R[l] / cfg.M_max;   // global radial cell spacing

        for (int m = 0; m < local_M_with_ghosts; ++m) {
            // m == 0 is the inner ghost column; its global index is m_start - 1.
            const int    m_global = m_start + m - 1;
            const double frac     = m_global * dy;   // == m_global / M_max

            r  [l][m] = (1.0 - frac) * geom.r_inner(z) + frac * geom.r_outer(z);
            r_z[l][m] = (1.0 - frac) * geom.dr_inner_dz(z)
                      + frac          * geom.dr_outer_dz(z);
        }
    }
}
