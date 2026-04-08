#pragma once

#include <vector>
#include "array2d.hpp"
#include "config.hpp"
#include "igeometry.hpp"

/// Stores the computational mesh for the axisymmetric coaxial channel.
///
/// With 2-D MPI decomposition both axes are local:
///   l ∈ [0 .. local_L_with_ghosts-1]  (includes ghost rows at 0 and local_L+1)
///   m ∈ [0 .. local_M_with_ghosts-1]  (includes ghost columns at 0 and local_M+1)
///
/// Coordinate mapping (using global indices):
///   m_global = m_start + m_local - 1   (m_local = 1 is the first owned cell)
///   r(l,m) = (1 - m_global*dy)*r_inner(z_l) + m_global*dy*r_outer(z_l)
///
/// The IGeometry reference must outlive this Grid object (both are typically
/// owned by main()).
class Grid {
public:
    const SimConfig&  cfg;
    const IGeometry&  geom;   ///< channel geometry — r_inner/r_outer profiles

    int local_L_with_ghosts;
    int l_start;   ///< global l index of the first owned interior cell

    int local_M_with_ghosts;
    int m_start;   ///< global m index of the first owned interior cell

    Array2D r;     ///< physical radial coordinate,  r[l_local][m_local]
    Array2D r_z;   ///< dr/dz at each node,           r_z[l_local][m_local]

    std::vector<double> R;   ///< R[l]  = r_outer(z) - r_inner(z),  radial span
    std::vector<double> dr;  ///< dr[l] = R[l] / M_max,             radial cell size

    /// Centreline radius at z = 0, used in boundary conditions.
    double r_0{};

    Grid(const SimConfig& cfg,
         int local_L_with_ghosts, int l_start,
         int local_M_with_ghosts, int m_start,
         const IGeometry& geom);

private:
    void Build();
};
