#pragma once

/**
 * @brief Numerically integrates a radially-symmetric function over a 2D
 * circular domain using the trapezoidal rule in polar coordinates.
 *
 * This function computes the integral of a function f(r) over a full circle (0
 * to 2π in angle) by applying the trapezoidal rule to the radial component. The
 * integration assumes the function is provided as discrete samples at equally
 * spaced radial points starting from r = 0.
 *
 * The integral computed is:
 * \f[
 *   \int_0^{2\pi} \int_0^{r_{\text{max}}} f(r) \, r \, dr \, d\theta = 2\pi
 * \int_0^{r_{\text{max}}} f(r) \, r \, dr
 * \f]
 *
 * where \f$ r_{\text{max}} = (n\_points - 1) \cdot dr \f$.
 *
 * @param func      Pointer to an array of function values f(r) sampled at
 * radial points. The array must contain at least `n_points` elements, where
 *                      `func[i]` corresponds to f(i * dr).
 * @param dr        Radial step size (spacing between consecutive sample
 * points).
 * @param n_points  Number of sample points in the `func` array. Must be >= 2.
 *
 * @return The approximate value of the 2D integral over the circular domain.
 *
 * @note The function assumes the input samples start at r = 0 and are uniformly
 * spaced by `dr`.
 * @note This implementation includes the Jacobian factor `r` from the polar
 * coordinate transformation and the full angular integration factor `2π`.
 *
 * @warning Behavior is undefined if `n_points < 2` or if `dr <= 0`.
 */
auto TrapezoidIntegrate(double *func, double dr, int n_points) -> double;

/**
 * Calculate the mass flux thoughout the right boundary of the nozzle
 * @param rho_last_col density on right boundary
 * @param v_z_last_col velocity on right boundary
 * @param dr_last_col  axial grid spacind on right boundary
 * @param M_max        number of points on the right boundaty
 * @return Mass flux thoughout the right boundary of the nozzle
 */
auto GetMassFlux(double *rho_last_col, double *v_z_last_col,
                     double dr_last_col, int M_max) -> double;

/**
 * @brief Computes the axial thrust using the stress tensor integrated over the
 * last radial column.
 *
 * This function calculates the thrust by evaluating the axial component of the
 * stress tensor at each radial grid point in the last column of the
 * computational domain and then integrating it using the trapezoidal rule
 *
 * @param rho_last_col      Array of density values at the last radial column
 * (size: M_max + 1).
 * @param v_z_last_col      Array of axial velocity components at the last
 * radial column (size: M_max + 1).
 * @param p_last_col        Array of gas pressure values at the last radial
 * column (size: M_max + 1).
 * @param H_r_last_col      Array of radial magnetic field components at the
 * last radial column (size: M_max + 1).
 * @param H_phi_last_col    Array of azimuthal magnetic field components at the
 * last radial column (size: M_max + 1).
 * @param H_z_last_col      Array of axial magnetic field components at the last
 * radial column (size: M_max + 1).
 * @param dr_last_col       Radial grid spacing at the last column (assumed
 * uniform).
 * @param M_max             Maximum radial index (number of radial zones minus
 * one).
 *
 * @return The integrated axial thrust value.
 */
auto GetThrust(double *rho_last_col, double *v_z_last_col,
                  double *p_last_col, double *H_r_last_col,
                  double *H_phi_last_col, double *H_z_last_col,
                  double dr_last_col, int M_max) -> double;
