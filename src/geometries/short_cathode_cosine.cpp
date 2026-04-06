#include "geometries/short_cathode_cosine.hpp"
#include <cmath>

// Geometry constants — all in the same normalised units as coaxial_nozzle.
namespace {
    constexpr double k_z_center  = 0.31;
    constexpr double k_width     = 0.015;
    constexpr double k_r_before  = 0.2;
    constexpr double k_r_after   = 0.005;
    constexpr double k_r_outer   = 0.8;

    constexpr double k_z_start   = k_z_center - k_width;   // 0.295
    constexpr double k_z_end     = k_z_center + k_width;   // 0.325
    constexpr double k_dr        = k_r_after - k_r_before;  // -0.195
    constexpr double k_inv_width = 1.0 / (2.0 * k_width);  // 1 / (z_end - z_start)
}

double ShortCathodeCosineGeometry::r_inner(double z) const {
    if (z < k_z_start)
        return k_r_before;
    if (z < k_z_end) {
        const double xi = (z - k_z_start) / (k_z_end - k_z_start);
        return k_r_before + k_dr * 0.5 * (1.0 - std::cos(M_PI * xi));
    }
    return k_r_after;
}

double ShortCathodeCosineGeometry::r_outer(double /*z*/) const {
    return k_r_outer;
}

double ShortCathodeCosineGeometry::dr_inner_dz(double z) const {
    if (z < k_z_start || z >= k_z_end)
        return 0.0;
    const double xi = (z - k_z_start) / (k_z_end - k_z_start);
    // d/dz [ ½(1 − cos(π·ξ)) ] = ½·π·sin(π·ξ) · dξ/dz
    return k_dr * 0.5 * M_PI * std::sin(M_PI * xi) * k_inv_width;
}

double ShortCathodeCosineGeometry::dr_outer_dz(double /*z*/) const {
    return 0.0;
}
