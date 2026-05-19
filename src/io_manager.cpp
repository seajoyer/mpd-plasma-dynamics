#include "io_manager.hpp"

#include <mpi.h>

#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridWriter.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

// ============================================================
// Constructor – create timestamped run directory and prepare
//               cached gather metadata.
// ============================================================

IOManager::IOManager(const SimConfig& cfg, const MPIManager& mpi) : cfg_(cfg), mpi_(mpi) {
    // Build a timestamp string on rank 0, then broadcast so every rank
    // ends up with the same directory name regardless of clock skew.
    char ts[64] = {};
    if (mpi_.rank == 0) {
        using namespace std::chrono;
        const auto now_tp = system_clock::now();
        const auto ms = duration_cast<milliseconds>(now_tp.time_since_epoch()) % 1000;
        const std::time_t tt = system_clock::to_time_t(now_tp);

        struct tm tm_info{};
        localtime_r(&tt, &tm_info);

        std::snprintf(ts, sizeof(ts), "%02d-%02d-%04d_%02d:%02d:%02d:%03d",
                      tm_info.tm_mday, tm_info.tm_mon + 1, tm_info.tm_year + 1900,
                      tm_info.tm_hour, tm_info.tm_min, tm_info.tm_sec,
                      static_cast<int>(ms.count()));
    }
    MPI_Bcast(ts, static_cast<int>(sizeof(ts)), MPI_CHAR, 0, MPI_COMM_WORLD);

    run_dir_ = cfg_.output_dir + "/" + cfg_.run_name + "_" + ts;

    if (mpi_.rank == 0) {
        std::error_code ec;
        std::filesystem::create_directories(run_dir_, ec);
        if (ec) {
            throw std::runtime_error("IOManager: cannot create run directory '" +
                                     run_dir_ + "': " + ec.message());
        }

        std::printf("Run directory: %s\n", run_dir_.c_str());
    }

    // All ranks wait until the directory is visible on the shared filesystem.
    MPI_Barrier(MPI_COMM_WORLD);

    // The MPI decomposition is fixed for the lifetime of this object, so the
    // gather metadata can be computed exactly once.
    BuildGatherMetadata();
}

// ============================================================
// Gather metadata — one-shot collective at construction.
//
// Each rank contributes its (l_start, m_start, local_L, local_M) tuple.
// Rank 0 gathers all of them in one MPI_Gather and pre-computes the
// recv-counts and recv-displacements needed by every subsequent
// MPI_Gatherv in WriteFrame().
// ============================================================

void IOManager::BuildGatherMetadata() {
    const int env_local[4] = {
        mpi_.l_start, mpi_.m_start, mpi_.local_L, mpi_.local_M
    };

    if (mpi_.rank == 0) {
        // Recv buffer for envelopes: 4 ints per rank.
        std::vector<int> env_all(static_cast<std::size_t>(mpi_.size) * 4);

        MPI_Gather(env_local, 4, MPI_INT,
                   env_all.data(), 4, MPI_INT,
                   /*root=*/0, MPI_COMM_WORLD);

        block_L_.resize(mpi_.size);
        block_M_.resize(mpi_.size);
        gl_     .resize(mpi_.size);
        gm_     .resize(mpi_.size);

        for (int r = 0; r < mpi_.size; ++r) {
            gl_     [r] = env_all[4 * r + 0];
            gm_     [r] = env_all[4 * r + 1];
            block_L_[r] = env_all[4 * r + 2];
            block_M_[r] = env_all[4 * r + 3];
        }

        // Pre-compute recv_counts / recv_displs for the per-frame
        // MPI_Gatherv.  Each rank contributes kNumFields × block_L × block_M
        // doubles, packed contiguously in field order.
        recv_counts_.resize(mpi_.size);
        recv_displs_.resize(mpi_.size);
        int running = 0;
        for (int r = 0; r < mpi_.size; ++r) {
            const int payload = kNumFields * block_L_[r] * block_M_[r];
            recv_counts_[r] = payload;
            recv_displs_[r] = running;
            running        += payload;
        }
        recv_buf_.resize(static_cast<std::size_t>(running));
    } else {
        // Non-root ranks: their pointer arguments to MPI_Gather are unused,
        // but the standard still requires the call to be made collectively.
        MPI_Gather(env_local, 4, MPI_INT,
                   /*recvbuf=*/nullptr, /*recvcount=*/0, MPI_INT,
                   /*root=*/0, MPI_COMM_WORLD);
    }
}

// ============================================================
// Public collective entry point
// ============================================================

void IOManager::WriteFrame(int step, const Fields& f, const Grid& grid) {
    // ── 1. Allocate global arrays on rank 0 (idempotent) ────────────────
    if (mpi_.rank == 0) {
        const int L_g = cfg_.L_max;
        const int M_g = cfg_.M_max + 1;  // global number of m nodes
        auto ensure = [&](Array2D& a) -> void {
            if (a.Rows() != L_g || a.Cols() != M_g) {
                a.Resize(L_g, M_g);
            }
        };
        ensure(rho_g_);
        ensure(v_z_g_);
        ensure(v_r_g_);
        ensure(v_phi_g_);
        ensure(e_g_);
        ensure(H_z_g_);
        ensure(H_r_g_);
        ensure(H_phi_g_);
        ensure(r_g_);
        ensure(rank_g_);
    }

    // ── 2. Pack this rank's interior blocks into send_buf_ ──────────────
    PackLocalBlocks(f, grid);

    // ── 3. Single MPI_Gatherv: every rank's payload onto rank 0 ─────────
    const int send_count = kNumFields * mpi_.local_L * mpi_.local_M;

    MPI_Gatherv(send_buf_.data(),       send_count,                 MPI_DOUBLE,
                (mpi_.rank == 0) ? recv_buf_.data()    : nullptr,
                (mpi_.rank == 0) ? recv_counts_.data() : nullptr,
                (mpi_.rank == 0) ? recv_displs_.data() : nullptr,
                MPI_DOUBLE,
                /*root=*/0, MPI_COMM_WORLD);

    // ── 4. Rank 0: unpack the ragged buffer + write the VTK file ────────
    if (mpi_.rank == 0) {
        UnpackGlobalBlocks();

        char filename[1024];
        std::snprintf(filename, sizeof(filename), "%s/step_%04d.vtk", run_dir_.c_str(),
                      step);
        WriteVtk(filename);
    }
}

// ============================================================
// PackLocalBlocks — flatten this rank's nine interior blocks
//                   into send_buf_ in canonical field order.
// ============================================================
//
// Layout in send_buf_  (interior cells only; ghost rows/cols skipped):
//
//   field 0 (rho)    : send_buf_[0                 .. 1×block      )
//   field 1 (v_z)    : send_buf_[1×block           .. 2×block      )
//   ...
//   field 8 (grid.r) : send_buf_[8×block           .. 9×block      )
//
// where block = local_L × local_M.  Within each field, cells are stored
// row-major in (l, m) with l outermost — same convention as Array2D.

void IOManager::PackLocalBlocks(const Fields& f, const Grid& grid) {
    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;
    const int block   = local_L * local_M;

    send_buf_.resize(static_cast<std::size_t>(kNumFields) * block);

    // Order must match the unpack side below.
    const Array2D* fields[kNumFields] = {
        &f.rho, &f.v_z, &f.v_r, &f.v_phi, &f.e,
        &f.H_z, &f.H_r, &f.H_phi, &grid.r
    };

    for (int fi = 0; fi < kNumFields; ++fi) {
        const Array2D& src = *fields[fi];
        double* dst = send_buf_.data() + static_cast<std::size_t>(fi) * block;
        for (int l = 0; l < local_L; ++l) {
            // Skip the ghost row at index 0 and the ghost column at index 0.
            const double* row = &src[l + 1][1];
            std::memcpy(dst + l * local_M, row, local_M * sizeof(double));
        }
    }
}

// ============================================================
// UnpackGlobalBlocks — rank-0 scatter from recv_buf_ into the
//                      nine global Array2D destinations.
// ============================================================
//
// Each rank r contributed kNumFields × block_L_[r] × block_M_[r] doubles
// starting at offset recv_displs_[r] in recv_buf_, with fields in the
// same canonical order used by PackLocalBlocks.

void IOManager::UnpackGlobalBlocks() {
    Array2D* global_fields[kNumFields] = {
        &rho_g_, &v_z_g_, &v_r_g_, &v_phi_g_, &e_g_,
        &H_z_g_, &H_r_g_, &H_phi_g_, &r_g_
    };

    for (int r = 0; r < mpi_.size; ++r) {
        const int gl       = gl_     [r];
        const int gm       = gm_     [r];
        const int block_L  = block_L_[r];
        const int block_M  = block_M_[r];
        const int block_sz = block_L * block_M;
        const double* base = recv_buf_.data() + recv_displs_[r];

        for (int fi = 0; fi < kNumFields; ++fi) {
            const double* src = base + static_cast<std::size_t>(fi) * block_sz;
            Array2D& dst = *global_fields[fi];
            for (int l = 0; l < block_L; ++l) {
                // One row of size block_M at a time → memcpy is well-defined
                // because Array2D's contiguous slab guarantees row[l][m...]
                // is a packed double array.
                std::memcpy(&dst[gl + l][gm], src + l * block_M,
                            block_M * sizeof(double));
            }
        }

        // Rank field: stamp the owning rank's id into rank_g_.  No comms
        // needed — block geometry is already in the cached metadata.
        if (cfg_.diagnostics.write_mpi_rank) {
            const double rank_id = static_cast<double>(r);
            for (int l = 0; l < block_L; ++l) {
                for (int m = 0; m < block_M; ++m) {
                    rank_g_[gl + l][gm + m] = rank_id;
                }
            }
        }
    }
}

// ============================================================
// VTK structured-grid write (rank 0 only)
// ============================================================

void IOManager::WriteVtk(const std::string& filepath) const {
    const int L_g = cfg_.L_max;
    const int M_max = cfg_.M_max;
    const int ni = L_g + 1;    // +1: VTK point count in z
    const int nj = M_max + 1;  // +1: VTK point count in r (= M_g nodes)

    // ---- points --------------------------------------------------------
    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataTypeToDouble();
    points->SetNumberOfPoints(static_cast<vtkIdType>(ni) * nj);

    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            const int li = std::min(i, L_g - 1);
            const double z = i * cfg_.dz;
            const double r = r_g_[li][j];
            // Convention: x = z (axial), y = r (radial), z_coord = 0.
            points->SetPoint(static_cast<vtkIdType>(i + j * ni), z, r, 0.0);
        }
    }

    // ---- structured grid -----------------------------------------------
    auto sg = vtkSmartPointer<vtkStructuredGrid>::New();
    sg->SetDimensions(ni, nj, 1);
    sg->SetPoints(points);

    // ---- helper: add scalar field --------------------------------------
    auto add_scalar = [&](const char* name, const Array2D& arr) -> void {
        auto da = vtkSmartPointer<vtkDoubleArray>::New();
        da->SetName(name);
        da->SetNumberOfTuples(static_cast<vtkIdType>(ni) * nj);
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                const int li = std::min(i, L_g - 1);
                da->SetValue(static_cast<vtkIdType>(i + j * ni), arr[li][j]);
            }
        }
        sg->GetPointData()->AddArray(da);
    };

    // ---- helper: add 3-component vector --------------------------------
    auto add_vector = [&](const char* name, const Array2D& az, const Array2D& ar,
                          const Array2D* aphi = nullptr) {
        auto da = vtkSmartPointer<vtkDoubleArray>::New();
        da->SetName(name);
        da->SetNumberOfComponents(3);
        da->SetNumberOfTuples(static_cast<vtkIdType>(ni) * nj);
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                const int li = std::min(i, L_g - 1);
                const double vphi = aphi ? (*aphi)[li][j] : 0.0;
                da->SetTuple3(static_cast<vtkIdType>(i + j * ni), az[li][j], ar[li][j],
                              vphi);
            }
        }
        sg->GetPointData()->AddArray(da);
    };

    // ---- scalar fields -------------------------------------------------
    add_scalar("Rho", rho_g_);
    add_scalar("Energy", e_g_);

    // ---- MPI domain decomposition visualisation ------------------------
    // Each cell is coloured by the rank that owns it.  In ParaView, apply
    // a "Surface" representation with the "MPI_Rank" array and a categorical
    // colour map to see individual subdomains at a glance.
    if (cfg_.diagnostics.write_mpi_rank) {
        add_scalar("MPI_Rank", rank_g_);
    }

    // ---- derived scalars -----------------------------------------------
    {
        auto hphi_r = vtkSmartPointer<vtkDoubleArray>::New();
        hphi_r->SetName("Hphi_times_r");
        hphi_r->SetNumberOfTuples(static_cast<vtkIdType>(ni) * nj);
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                const int li = std::min(i, L_g - 1);
                hphi_r->SetValue(static_cast<vtkIdType>(i + j * ni),
                                 H_phi_g_[li][j] * r_g_[li][j]);
            }
        }
        sg->GetPointData()->AddArray(hphi_r);
    }

    // ---- vector fields -------------------------------------------------
    add_vector("Velocity", v_z_g_, v_r_g_, &v_phi_g_);
    add_vector("MagneticField", H_z_g_, H_r_g_, &H_phi_g_);

    // ---- write ---------------------------------------------------------
    auto writer = vtkSmartPointer<vtkStructuredGridWriter>::New();
    writer->SetFileName(filepath.c_str());
    writer->SetInputData(sg);
    writer->SetFileTypeToBinary();
    writer->Write();

    if (writer->GetErrorCode() != 0) {
        std::cerr << "Warning: VTK writer reported an error for " << filepath << "\n";
    }
}
