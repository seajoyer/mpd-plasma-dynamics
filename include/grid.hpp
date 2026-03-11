#pragma once

#include <vector>
#include "array2d.hpp"
#include "config.hpp"

/// Stores the computational mesh for the axisymmetric coaxial channel.
///
/// The grid maps from logical index (l, m) to physical coordinates (z, r).
/// z runs along the axis (l-direction, parallelised with MPI).
/// r runs radially from the inner wall r1(z) to the outer wall r2(z).
///
/// Coordinate mapping per column:
///   r(l,m) = (1 - m*dy) * r1(z_l)  +  m*dy * r2(z_l)
class Grid {
public:
    const SimConfig& cfg;

    int local_L_with_ghosts;
    int l_start;   ///< global l index of the first owned cell (before ghost offset)

    Array2D r;     ///< physical radial coordinate r[l][m]
    Array2D r_z;   ///< dr/dz at each node, r_z[l][m]

    std::vector<double> R;   ///< R[l]  = r2(z) - r1(z), radial span
    std::vector<double> dr;  ///< dr[l] = R[l] / M_max,  radial cell size

    /// Centreline radius at z = 0, used in boundary conditions.
    double r_0{};

    Grid(const SimConfig& cfg, int local_L_with_ghosts, int l_start);

    // ---- static geometry functions (public so BCs can call them) ----
    static double r1(double z);
    static double r2(double z);
    static double der_r1(double z);
    static double der_r2(double z);

private:
    void build();
};
