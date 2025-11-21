#include "geometry.hpp"
#include <cmath>

auto R1(double z) -> double {
    const double z_center = 0.31;      // Центр ступеньки
    const double transition_width = 0.015; // Полуширина перехода (уже в ~5 раз)
    const double r_before = 0.2;       // Радиус до ступеньки
    const double r_after = 0.005;      // Радиус после ступеньки (узкое горло)

    // Начало и конец переходной зоны
    const double z_start = z_center - transition_width;
    const double z_end = z_center + transition_width;

    if (z < z_start) {
        // Постоянный радиус до ступеньки
        return r_before;
    }
    else if (z >= z_start && z < z_end) {
        // Плавный переход используя косинусоидальную функцию (C² непрерывность)
        // Это обеспечивает гладкость производных
        double xi = (z - z_start) / (z_end - z_start); // нормализованная координата [0,1]
        double smooth_factor = 0.5 * (1.0 - cos(M_PI * xi)); // S-образная кривая
        return r_before + (r_after - r_before) * smooth_factor;
    }
    else {
        return r_after;
    }
}


auto R2(double z) -> double {
    return 0.8;
}

auto DerR1(double z) -> double {
    const double z_center = 0.31;
    const double transition_width = 0.015;
    const double r_before = 0.2;
    const double r_after = 0.005;

    const double z_start = z_center - transition_width;
    const double z_end = z_center + transition_width;

    if (z < z_start) {
        return 0.0; // Постоянный радиус - нулевая производная
    }
    else if (z >= z_start && z < z_end) {
        // Производная косинусоидальной функции
        double xi = (z - z_start) / (z_end - z_start);
        double dxi_dz = 1.0 / (z_end - z_start);
        double dsmooth_dxi = 0.5 * M_PI * sin(M_PI * xi);
        return (r_after - r_before) * dsmooth_dxi * dxi_dz;
    }
    else {
        return 0.0; // Постоянный радиус после ступеньки
    }
}

auto DerR2(double z) -> double {
    return 0.0;
}
