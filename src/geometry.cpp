#include "geometry.hpp"
#include <cmath>

auto R1(double z) -> double {
    if (z < 0.3) {
        return 0.2;
    } else if (z >= 0.3 && z < 0.4) {
        return 0.2 - 10 * pow((z - 0.3), 2);
    } else if (z >= 0.4 && z < 0.478) {
        return 10 * pow((z - 0.5), 2);
    } else {
        return 0.005;
    }
}

auto R2(double z) -> double {
    return 0.8;
}

auto DerR1(double z) -> double {
    if (z < 0.3) {
        return 0.0;
    } else if (z >= 0.3 && z < 0.4) {
        return -10 * 2 * (z - 0.3);
    } else if (z >= 0.4 && z < 0.478) {
        return 10 * 2 * (z - 0.5);
    } else {
        return 0.0;
    }
}

auto DerR2(double z) -> double {
    return 0.0;
}
