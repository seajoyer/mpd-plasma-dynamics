#include "verification.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "array2d.hpp"

namespace Verification {

// ─── 1. Conserved integrals ──────────────────────────────────────────────

auto ComputeIntegrals(const Fields& f, const Grid& g,
                                    const SimConfig& cfg,
                                    const MPIManager& mpi) -> ConservedIntegrals {
    const double dz = cfg.dz;
    double local[7] = {};

    for (int l = 1; l <= mpi.local_L; ++l) {
        const double cell_area = g.dr[l] * dz;
        for (int m = 1; m <= mpi.local_M; ++m) {
            local[0] += f.u0_1[l][m] * cell_area;
            local[1] += f.u0_2[l][m] * cell_area;
            local[2] += f.u0_3[l][m] * cell_area;
            local[3] += f.u0_4[l][m] * cell_area;
            local[4] += f.u0_5[l][m] * cell_area;
            local[5] += f.u0_7[l][m] * cell_area;
            local[6] += f.u0_6[l][m] * cell_area;
        }
    }

    double global[7] = {};
    MPI_Allreduce(local, global, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return {.mass         = global[0],
            .momentum_z   = global[1],
            .momentum_r   = global[2],
            .momentum_phi = global[3],
            .energy       = global[4],
            .mag_flux_z   = global[5],
            .mag_flux_phi = global[6]};
}

void PrintIntegralsHeader(int rank) {
    if (rank != 0) return;
    std::printf("\n[Verification] Conserved integrals\n");
    std::printf("  %-8s  %-12s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s\n",
                "step", "t",
                "mass", "mom_z", "mom_r", "mom_phi",
                "energy", "mag_z", "mag_phi");
    std::printf("  %-8s  %-12s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s\n",
                "--------", "------------",
                "--------------", "--------------", "--------------", "--------------",
                "--------------", "--------------", "--------------");
}

void PrintIntegrals(const ConservedIntegrals& q, int step, double t, int rank) {
    if (rank != 0) return;
    std::printf(
        "  %-8d  %-12.5f  %-14.6e  %-14.6e  %-14.6e  %-14.6e  %-14.6e  %-14.6e  %-14.6e\n",
        step, t,
        q.mass, q.momentum_z, q.momentum_r, q.momentum_phi,
        q.energy, q.mag_flux_z, q.mag_flux_phi);
}

auto ReportDrift(const ConservedIntegrals& ref,
                   const ConservedIntegrals& cur,
                   int step, double t, int rank) -> double {
    constexpr double eps = 1.0e-30;

    const double ref_vals[7] = {
        ref.mass, ref.momentum_z, ref.momentum_r, ref.momentum_phi,
        ref.energy, ref.mag_flux_z, ref.mag_flux_phi
    };
    const double cur_vals[7] = {
        cur.mass, cur.momentum_z, cur.momentum_r, cur.momentum_phi,
        cur.energy, cur.mag_flux_z, cur.mag_flux_phi
    };
    const char* const names[7] = {
        "mass", "mom_z", "mom_r", "mom_phi", "energy", "mag_z", "mag_phi"
    };

    double scale = 0.0;
    for (double ref_val : ref_vals) scale = std::max(scale, std::abs(ref_val));
    if (scale < 1e-30) {
        for (double cur_val : cur_vals) scale = std::max(scale, std::abs(cur_val));
    }
    if (scale < 1e-30) scale = 1.0;  // all zeros — degenerate case

    double max_drift = 0.0;
    for (int i = 0; i < 7; ++i) {
        const double drift = std::abs(cur_vals[i] - ref_vals[i]) / scale;
        max_drift = std::max(max_drift, drift);
    }

    if (rank == 0) {
        std::printf("[Verification] Drift from step-0 reference"
                    " at step %d (t=%.5f)\n", step, t);
        std::printf("  %-10s  %-18s  %-18s  %-12s\n",
                    "quantity", "reference", "current", "rel_drift");
        std::printf("  %-10s  %-18s  %-18s  %-12s\n",
                    "----------", "------------------",
                    "------------------", "------------");
        for (int i = 0; i < 7; ++i) {
            const double drift = std::abs(cur_vals[i] - ref_vals[i])
                               / (std::abs(ref_vals[i]) + eps);
            std::printf("  %-10s  %+18.8e  %+18.8e  %12.4e\n",
                        names[i], ref_vals[i], cur_vals[i], drift);
        }
        std::printf("  max_drift = %.4e\n\n", max_drift);
    }

    return max_drift;
}

// ─── 2. Radial-symmetry check ────────────────────────────────────────────

auto CheckRadialSymmetry(const Fields& f, const MPIManager& mpi) -> double {
    double local_max = 0.0;

    for (int l = 1; l <= mpi.local_L; ++l) {
        for (int m = 1; m < mpi.local_M; ++m) {
            const double a    = f.rho[l][m];
            const double b    = f.rho[l][m + 1];
            const double mean = 0.5 * (a + b);
            if (mean > 1.0e-15) {
                local_max = std::max(local_max, std::abs(b - a) / mean);
            }
        }
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (mpi.rank == 0) {
        const bool uniform = (global_max < 1.0e-10);
        std::printf("[Verification] Radial symmetry : max |Δρ|/<ρ> = %.3e  (%s)\n",
                    global_max,
                    uniform ? "field is radially uniform" : "radial variation present");
    }

    return global_max;
}

// ─── 3. Ghost-exchange round-trip ────────────────────────────────────────

auto CheckGhostExchange(MPIManager& mpi, const SimConfig& cfg) -> bool {
    const int lwg = mpi.local_L_with_ghosts;
    const int mwg = mpi.local_M_with_ghosts;

    Array2D arr(lwg, mwg);   // zero-initialised by Array2D::Allocate

    // Encode interior cell (l_local, m_local) as a unique positive double:
    //   f(l, m) = (l_global + 1) * stride + (m_global + 1)
    //
    // The +1 offsets ensure no interior cell is ever encoded as 0, making
    // un-exchanged ghost cells (which remain at 0) trivially detectable.
    // stride = M_max + 10 prevents collisions between cells in different rows.
    const int stride = cfg.M_max + 10;

    for (int l = 1; l <= mpi.local_L; ++l) {
        const int lg = mpi.l_start + l - 1;   // global l index
        for (int m = 1; m <= mpi.local_M; ++m) {
            const int mg = mpi.m_start + m - 1;   // global m index
            arr[l][m] = static_cast<double>((lg + 1) * stride + (mg + 1));
        }
    }

    // Run the exact same exchange path the solver uses.
    std::vector<double> bufs;
    double** ptrs[1] = { arr.Raw() };
    mpi.ExchangeGhostsBatch(ptrs, 1, bufs);

    // Verify received ghost cells.
    //
    // For each active neighbour direction the sending rank's interior row/column
    // is known from the adjacency of the Cartesian decomposition:
    //
    //   l=0 ghost        ← sending rank's last  interior row  (l_global = l_start − 1)
    //   l=local_L+1 ghost← sending rank's first interior row  (l_global = l_end   + 1)
    //   m=0 ghost        ← sending rank's last  interior col  (m_global = m_start − 1)
    //   m=local_M+1 ghost← sending rank's first interior col  (m_global = m_end   + 1)
    //
    // Corner cells (where a ghost row meets a ghost column) are skipped because
    // the sending rank never filled them (they were 0 before its own exchange).
    int local_errors = 0;

    // l = 0 ghost row: interior m-columns only
    if (mpi.nbr_l_lo != MPI_PROC_NULL) {
        const int lg_send = mpi.l_start - 1;
        for (int m = 1; m <= mpi.local_M; ++m) {
            const int mg = mpi.m_start + m - 1;
            const double expected = static_cast<double>((lg_send + 1) * stride + (mg + 1));
            if (std::abs(arr[0][m] - expected) > 0.5) { ++local_errors; }
        }
    }

    // l = local_L + 1 ghost row: interior m-columns only
    if (mpi.nbr_l_hi != MPI_PROC_NULL) {
        const int lg_send = mpi.l_end + 1;
        for (int m = 1; m <= mpi.local_M; ++m) {
            const int mg = mpi.m_start + m - 1;
            const double expected = static_cast<double>((lg_send + 1) * stride + (mg + 1));
            if (std::abs(arr[mpi.local_L + 1][m] - expected) > 0.5) { ++local_errors; }
        }
    }

    // m = 0 ghost column: interior l-rows only
    if (mpi.nbr_m_lo != MPI_PROC_NULL) {
        const int mg_send = mpi.m_start - 1;
        for (int l = 1; l <= mpi.local_L; ++l) {
            const int lg = mpi.l_start + l - 1;
            const double expected = static_cast<double>((lg + 1) * stride + (mg_send + 1));
            if (std::abs(arr[l][0] - expected) > 0.5) { ++local_errors; }
        }
    }

    // m = local_M + 1 ghost column: interior l-rows only
    if (mpi.nbr_m_hi != MPI_PROC_NULL) {
        const int mg_send = mpi.m_end + 1;
        for (int l = 1; l <= mpi.local_L; ++l) {
            const int lg = mpi.l_start + l - 1;
            const double expected = static_cast<double>((lg + 1) * stride + (mg_send + 1));
            if (std::abs(arr[l][mpi.local_M + 1] - expected) > 0.5) { ++local_errors; }
        }
    }

    int global_errors = 0;
    MPI_Allreduce(&local_errors, &global_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (mpi.rank == 0) {
        if (global_errors == 0) {
            std::printf("[Verification] Ghost exchange  : PASS\n");
        } else {
            std::printf("[Verification] Ghost exchange  : FAIL"
                        "  (%d incorrect ghost cell(s) across all ranks)\n",
                        global_errors);
        }
    }

    return (global_errors == 0);
}

} // namespace Verification
