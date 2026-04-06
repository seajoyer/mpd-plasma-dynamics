#include "geometries/short_cathode_parabolic.hpp"
#include <cmath>

double ShortCathodeParabolicGeometry::r_inner(double z) const {
    if      (z < 0.3)   return 0.2;
    else if (z < 0.4)   return 0.2 - 10.0 * std::pow(z - 0.3, 2);
    else if (z < 0.478) return 10.0 * std::pow(z - 0.5, 2);
    else                return 0.005;
}

double ShortCathodeParabolicGeometry::r_outer(double /*z*/) const {
    return 0.8;
}

double ShortCathodeParabolicGeometry::dr_inner_dz(double z) const {
    if      (z < 0.3)   return  0.0;
    else if (z < 0.4)   return -20.0 * (z - 0.3);
    else if (z < 0.478) return  20.0 * (z - 0.5);
    else                return  0.0;
}

double ShortCathodeParabolicGeometry::dr_outer_dz(double /*z*/) const {
    return 0.0;
}
