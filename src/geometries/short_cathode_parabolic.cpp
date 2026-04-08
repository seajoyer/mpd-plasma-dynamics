#include "geometries/short_cathode_parabolic.hpp"
#include <cmath>

auto ShortCathodeParabolicGeometry::RInner(double z) const -> double {
    if        (z < 0.3)   { return 0.2;
    } else if (z < 0.4)   { return 0.2 - 10.0 * std::pow(z - 0.3, 2);
    } else if (z < 0.478) { return 10.0 * std::pow(z - 0.5, 2);
    } else {                return 0.005;
}
}

auto ShortCathodeParabolicGeometry::ROuter(double /*z*/) const -> double {
    return 0.8;
}

auto ShortCathodeParabolicGeometry::DrInnerDz(double z) const -> double {
    if        (z < 0.3)   { return  0.0;
    } else if (z < 0.4)   { return -20.0 * (z - 0.3);
    } else if (z < 0.478) { return  20.0 * (z - 0.5);
    } else {                return  0.0;
}
}

auto ShortCathodeParabolicGeometry::DrOuterDz(double /*z*/) const -> double {
    return 0.0;
}
