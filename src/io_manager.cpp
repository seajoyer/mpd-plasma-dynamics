#include "io_manager.hpp"

#include <mpi.h>

// VTK headers
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridWriter.h>

#include <algorithm>
#include <chrono>
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

IOManager::IOManager(const SimConfig& cfg, const MPIManager& mpi)
    : cfg_(cfg), mpi_(mpi)
{
    // Build a timestamp string on rank 0, then broadcast so every rank
    // ends up with the same directory name regardless of clock skew.
    char ts[64] = {};
    if (mpi_.rank == 0) {
        using namespace std::chrono;
        const auto now_tp = system_clock::now();
        const auto ms     = duration_cast<milliseconds>(now_tp.time_since_epoch()) % 1000;
        const std::time_t tt = system_clock::to_time_t(now_tp);

        struct tm tm_info{};
        localtime_r(&tt, &tm_info);

        std::snprintf(ts, sizeof(ts),
                      "%02d-%02d-%04d_%02d:%02d:%02d:%03d",
                      tm_info.tm_mday,
                      tm_info.tm_mon + 1,
                      tm_info.tm_year + 1900,
                      tm_info.tm_hour,
                      tm_info.tm_min,
                      tm_info.tm_sec,
                      static_cast<int>(ms.count()));
    }
    MPI_Bcast(ts, static_cast<int>(sizeof(ts)), MPI_CHAR, 0, MPI_COMM_WORLD);

    run_dir_ = cfg_.output_dir + "/" + cfg_.run_name + "_" + ts;

    if (mpi_.rank == 0) {
        std::error_code ec;
        std::filesystem::create_directories(run_dir_, ec);
        if (ec)
            throw std::runtime_error("IOManager: cannot create run directory '"
                                     + run_dir_ + "': " + ec.message());

        std::printf("Run directory: %s\n", run_dir_.c_str());
    }

    // All ranks wait until the directory is visible on the shared file system.
    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================
// Public collective entry point
// ============================================================

void IOManager::write_frame(int step, const Fields& f, const Grid& grid) {
    gather_global(f, grid);

    if (mpi_.rank == 0) {
        char filename[1024];
        std::snprintf(filename, sizeof(filename),
                      "%s/step_%04d.vtk", run_dir_.c_str(), step);
        write_vtk(filename);
    }
}

// ============================================================
// Gather local → global (collective)
// ============================================================

void IOManager::gather_global(const Fields& f, const Grid& grid) {
    const int L_g   = cfg_.L_max_global;
    const int M_max = cfg_.M_max;

    // Allocate (or reuse if already the right size) on rank 0.
    if (mpi_.rank == 0) {
        auto ensure = [&](Array2D& a) {
            if (a.rows() != L_g || a.cols() != M_max + 1)
                a.resize(L_g, M_max + 1);
        };
        ensure(rho_g_);  ensure(v_z_g_);  ensure(v_r_g_);
        ensure(v_phi_g_); ensure(e_g_);
        ensure(H_z_g_);  ensure(H_r_g_);  ensure(H_phi_g_);
        ensure(r_g_);
    }

    const int local_L = mpi_.local_L;

    // Build recvcounts / displs once (same layout every call).
    std::vector<int> recvcounts(mpi_.size), displs(mpi_.size);
    for (int i = 0; i < mpi_.size; ++i) {
        const int is = i * mpi_.L_per_proc;
        int       ie = (i + 1) * mpi_.L_per_proc - 1;
        if (i == mpi_.size - 1) ie = L_g - 1;
        recvcounts[i] = ie - is + 1;
        displs[i]     = is;
    }

    std::vector<double> local_row(local_L);
    std::vector<double> global_row(mpi_.rank == 0 ? L_g : 0);

    // Gather one 2-D field, column-by-column along m.
    auto gather_field = [&](const Array2D& src, Array2D& dst) {
        for (int m = 0; m < M_max + 1; ++m) {
            for (int l = 0; l < local_L; ++l)
                local_row[l] = src[l + 1][m];   // l+1 skips left ghost cell

            MPI_Gatherv(local_row.data(), local_L, MPI_DOUBLE,
                        global_row.data(), recvcounts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (mpi_.rank == 0)
                for (int l = 0; l < L_g; ++l)
                    dst[l][m] = global_row[l];
        }
    };

    gather_field(f.rho,   rho_g_);
    gather_field(f.v_z,   v_z_g_);
    gather_field(f.v_r,   v_r_g_);
    gather_field(f.v_phi, v_phi_g_);
    gather_field(f.e,     e_g_);
    gather_field(f.H_z,   H_z_g_);
    gather_field(f.H_r,   H_r_g_);
    gather_field(f.H_phi, H_phi_g_);
    gather_field(grid.r,  r_g_);
}

// ============================================================
// VTK structured-grid write (rank 0 only)
// ============================================================

void IOManager::write_vtk(const std::string& filepath) const {
    // Grid dimensions:
    //   ni = L_g + 1  points in the z (axial) direction
    //   nj = M_max+1  points in the r (radial) direction
    //   nk = 1        (2-D slice)
    //
    // The gathered arrays have L_g rows (indices 0 … L_g-1).
    // The (L_g+1)-th z-plane is filled by repeating the last row,
    // which matches the treatment used in the original writer.

    const int L_g   = cfg_.L_max_global;
    const int M_max = cfg_.M_max;
    const int ni    = L_g + 1;
    const int nj    = M_max + 1;

    // ---- points --------------------------------------------------------
    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataTypeToDouble();
    points->SetNumberOfPoints(static_cast<vtkIdType>(ni) * nj);

    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            const int    li = std::min(i, L_g - 1);
            const double z  = i * cfg_.dz;
            const double r  = r_g_[li][j];
            // Convention: x = z (axial), y = r (radial), z = 0.
            points->SetPoint(static_cast<vtkIdType>(i + j * ni), z, r, 0.0);
        }
    }

    // ---- structured grid -----------------------------------------------
    auto sg = vtkSmartPointer<vtkStructuredGrid>::New();
    sg->SetDimensions(ni, nj, 1);
    sg->SetPoints(points);

    // ---- helper lambdas ------------------------------------------------

    // Add a scalar point-data array from an Array2D.
    auto add_scalar = [&](const char* name, const Array2D& arr) {
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

    // Add a 3-component vector from two in-plane arrays + optional phi component.
    auto add_vector = [&](const char* name,
                          const Array2D& az, const Array2D& ar,
                          const Array2D* aphi = nullptr) {
        auto da = vtkSmartPointer<vtkDoubleArray>::New();
        da->SetName(name);
        da->SetNumberOfComponents(3);
        da->SetNumberOfTuples(static_cast<vtkIdType>(ni) * nj);
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                const int    li   = std::min(i, L_g - 1);
                const double vphi = aphi ? (*aphi)[li][j] : 0.0;
                da->SetTuple3(static_cast<vtkIdType>(i + j * ni),
                              az[li][j], ar[li][j], vphi);
            }
        }
        sg->GetPointData()->AddArray(da);
    };

    // ---- scalar fields -------------------------------------------------
    add_scalar("Rho",    rho_g_);
    add_scalar("Energy", e_g_);
    add_scalar("Vphi",   v_phi_g_);
    add_scalar("Hphi",   H_phi_g_);

    // ---- derived scalars -----------------------------------------------
    {
        auto vl = vtkSmartPointer<vtkDoubleArray>::New();
        vl->SetName("SpeedInPlane");
        vl->SetNumberOfTuples(static_cast<vtkIdType>(ni) * nj);
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                const int li = std::min(i, L_g - 1);
                const double v = std::sqrt(v_z_g_[li][j] * v_z_g_[li][j]
                                         + v_r_g_[li][j] * v_r_g_[li][j]);
                vl->SetValue(static_cast<vtkIdType>(i + j * ni), v);
            }
        }
        sg->GetPointData()->AddArray(vl);
    }
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
    add_vector("Velocity",       v_z_g_, v_r_g_, &v_phi_g_);
    add_vector("MagneticField",  H_z_g_, H_r_g_, &H_phi_g_);

    // ---- write ---------------------------------------------------------
    auto writer = vtkSmartPointer<vtkStructuredGridWriter>::New();
    writer->SetFileName(filepath.c_str());
    writer->SetInputData(sg);
    writer->SetFileTypeToBinary();   // compact; use SetFileTypeToASCII() if needed
    writer->Write();

    if (writer->GetErrorCode() != 0)
        std::cerr << "Warning: VTK writer reported an error for " << filepath << "\n";
}
