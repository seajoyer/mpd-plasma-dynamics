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
// Constructor – create timestamped run directory
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
}

// ============================================================
// Public collective entry point
// ============================================================

void IOManager::WriteFrame(int step, const Fields& f, const Grid& grid) {
    GatherGlobal(f, grid);

    if (mpi_.rank == 0) {
        char filename[1024];
        std::snprintf(filename, sizeof(filename), "%s/step_%04d.vtk", run_dir_.c_str(),
                      step);
        WriteVtk(filename);
    }
}

// ============================================================
// Static helpers: pack / unpack / fill-rank
// ============================================================

void IOManager::PackField(const Array2D& src, int local_L, int local_M,
                           std::vector<double>& buf) {
    // Row-major: outer index l, inner index m.
    buf.resize(local_L * local_M);
    for (int l = 0; l < local_L; ++l) {
        for (int m = 0; m < local_M; ++m) {
            buf[l * local_M + m] = src[l + 1][m + 1];  // skip ghost rows/cols
        }
    }
}

void IOManager::UnpackIntoGlobal(Array2D& dst, const std::vector<double>& buf, int gl,
                                   int gm, int block_L, int block_M) {
    for (int l = 0; l < block_L; ++l) {
        for (int m = 0; m < block_M; ++m) {
            dst[gl + l][gm + m] = buf[l * block_M + m];
        }
    }
}

void IOManager::FillRankBlock(Array2D& dst, double rank_id, int gl, int gm, int block_L,
                                int block_M) {
    // No data transfer needed — rank 0 already knows which block belongs to
    // which rank from the gather envelope, so we just stamp the value.
    for (int l = 0; l < block_L; ++l) {
        for (int m = 0; m < block_M; ++m) {
            dst[gl + l][gm + m] = rank_id;
        }
    }
}

// ============================================================
// Point-to-point gather — all ranks participate
// ============================================================

void IOManager::GatherGlobal(const Fields& f, const Grid& grid) {
    const int L_g = cfg_.L_max;
    const int M_g = cfg_.M_max + 1;  // global number of m nodes
    const int nfields = 9;           // rho, v_z, v_r, v_phi, e, H_z, H_r, H_phi, r

    // ---------- rank 0: allocate global arrays once ----------------------
    if (mpi_.rank == 0) {
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

    // ---------- message tag layout: tag = field_index (0-8) --------------
    // Each non-zero rank sends one envelope + (nfields) data messages.
    //
    // Envelope layout (4 ints):  l_start, m_start, local_L, local_M
    //   where l_start / m_start are the global indices of the first owned
    //   interior cell.

    const int local_L = mpi_.local_L;
    const int local_M = mpi_.local_M;

    // The array list must match the order expected by the unpack loop below.
    const Array2D* local_fields[nfields] = {&f.rho, &f.v_z, &f.v_r,   &f.v_phi, &f.e,
                                            &f.H_z, &f.H_r, &f.H_phi, &grid.r};

    Array2D* global_fields[nfields] = {&rho_g_, &v_z_g_, &v_r_g_,   &v_phi_g_, &e_g_,
                                       &H_z_g_, &H_r_g_, &H_phi_g_, &r_g_};

    // ---------- non-zero ranks: pack and send ----------------------------
    if (mpi_.rank != 0) {
        // Envelope
        int env[4] = {mpi_.l_start, mpi_.m_start, local_L, local_M};
        MPI_Send(env, 4, MPI_INT, 0, /*tag=*/100, MPI_COMM_WORLD);

        // Field data
        std::vector<double> buf;
        for (int fi = 0; fi < nfields; ++fi) {
            PackField(*local_fields[fi], local_L, local_M, buf);
            MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, 0, fi,
                     MPI_COMM_WORLD);
        }
        return;
    }

    // ---------- rank 0: copy own block directly --------------------------
    {
        const int gl = mpi_.l_start;  // == 0
        const int gm = mpi_.m_start;  // == 0
        std::vector<double> buf;
        for (int fi = 0; fi < nfields; ++fi) {
            PackField(*local_fields[fi], local_L, local_M, buf);
            UnpackIntoGlobal(*global_fields[fi], buf, gl, gm, local_L, local_M);
        }
        // Rank field: no send/recv needed — stamp rank 0's block directly.
        FillRankBlock(rank_g_, 0.0, gl, gm, local_L, local_M);
    }

    // ---------- rank 0: receive from all other ranks ---------------------
    std::vector<double> recv_buf;
    for (int src = 1; src < mpi_.size; ++src) {
        // Receive envelope.
        // MPI_STATUS_IGNORE (singular) is the correct handle for a single MPI_Recv.
        int env[4];
        MPI_Recv(env, 4, MPI_INT, src, /*tag=*/100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        const int gl = env[0];
        const int gm = env[1];
        const int block_L = env[2];
        const int block_M = env[3];
        const int buf_size = block_L * block_M;

        recv_buf.resize(buf_size);
        for (int fi = 0; fi < nfields; ++fi) {
            MPI_Recv(recv_buf.data(), buf_size, MPI_DOUBLE, src, fi, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            UnpackIntoGlobal(*global_fields[fi], recv_buf, gl, gm, block_L, block_M);
        }
        // Rank field: the envelope already tells us the owning rank — no
        // extra message required.
        FillRankBlock(rank_g_, static_cast<double>(src), gl, gm, block_L, block_M);
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
    add_scalar("MPI_Rank", rank_g_);

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
