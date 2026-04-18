#include "ics/vtk_ic.hpp"

#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridReader.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"

// ============================================================
// Impl — stores the complete field data read from the VTK file
// ============================================================

struct VtkIC::Impl {
    // VTK grid dimensions: ni = L_max_old + 1,  nj = M_max_old + 1.
    int ni{};
    int nj{};

    // Flat field arrays, indexed as  k = i + j * ni
    // where i is the axial (z) point index and j the radial (r) index.
    // These correspond exactly to the VTK point-data arrays written by
    // IOManager::WriteVtk, so  data[i + j*ni] = field value at global
    // simulation cell (i, j) for i < L_max_old.
    std::vector<double> rho;
    std::vector<double> v_z, v_r, v_phi;
    std::vector<double> H_z, H_r, H_phi;
    std::vector<double> e;

    // ----------------------------------------------------------------
    // Constructor — opens the file and populates all field vectors.
    // ----------------------------------------------------------------
    explicit Impl(const std::string& path) {
        // Check existence before handing off to VTK so the error message
        // is clear even on platforms where VTK swallows file-open errors.
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error(
                "VtkIC: file not found: '" + path + "'\n"
                "  Check that the path is correct and accessible from every "
                "MPI rank (shared filesystem required).");
        }

        // ---- Read the file ----
        auto reader = vtkSmartPointer<vtkStructuredGridReader>::New();
        reader->SetFileName(path.c_str());
        reader->ReadAllScalarsOn();
        reader->ReadAllVectorsOn();
        reader->Update();

        vtkStructuredGrid* sg = reader->GetOutput();
        if (!sg || sg->GetNumberOfPoints() == 0) {
            throw std::runtime_error(
                "VtkIC: could not read VTK structured grid from '" + path + "'.\n"
                "  The file may be corrupt or not in legacy structured-grid format.");
        }

        // ---- Grid dimensions ----
        int dims[3]{1, 1, 1};
        sg->GetDimensions(dims);
        ni = dims[0];   // axial   (= L_max_old + 1)
        nj = dims[1];   // radial  (= M_max_old + 1)

        if (ni < 2 || nj < 2) {
            throw std::runtime_error(
                "VtkIC: VTK grid too small (" + std::to_string(ni) + " × " +
                std::to_string(nj) + ").  Expected at least 2 × 2.");
        }

        const auto n_pts = static_cast<vtkIdType>(ni) * nj;

        // ---- Helper: extract a scalar array into a std::vector<double> ----
        auto read_scalar = [&](const char* name) -> std::vector<double> {
            vtkDataArray* arr = sg->GetPointData()->GetArray(name);
            if (!arr) {
                throw std::runtime_error(
                    std::string("VtkIC: point-data array '") + name +
                    "' not found in '" + path + "'.\n"
                    "  The file must have been written by IOManager::WriteFrame.");
            }
            std::vector<double> out(n_pts);
            for (vtkIdType k = 0; k < n_pts; ++k) {
                out[static_cast<std::size_t>(k)] = arr->GetComponent(k, 0);
            }
            return out;
        };

        // ---- Helper: extract one component of a multi-component array ----
        auto read_component = [&](const char* name, int comp) -> std::vector<double> {
            vtkDataArray* arr = sg->GetPointData()->GetArray(name);
            if (!arr) {
                throw std::runtime_error(
                    std::string("VtkIC: point-data array '") + name +
                    "' not found in '" + path + "'.\n"
                    "  The file must have been written by IOManager::WriteFrame.");
            }
            if (arr->GetNumberOfComponents() <= comp) {
                throw std::runtime_error(
                    std::string("VtkIC: array '") + name + "' has only " +
                    std::to_string(arr->GetNumberOfComponents()) +
                    " component(s); requested component " + std::to_string(comp) + ".");
            }
            std::vector<double> out(n_pts);
            for (vtkIdType k = 0; k < n_pts; ++k) {
                out[static_cast<std::size_t>(k)] = arr->GetComponent(k, comp);
            }
            return out;
        };

        // ---- Extract all eight physical fields ----
        rho   = read_scalar("Rho");
        e     = read_scalar("Energy");
        v_z   = read_component("Velocity",      0);
        v_r   = read_component("Velocity",      1);
        v_phi = read_component("Velocity",      2);
        H_z   = read_component("MagneticField", 0);
        H_r   = read_component("MagneticField", 1);
        H_phi = read_component("MagneticField", 2);
    }

    // ----------------------------------------------------------------
    // Bilinear interpolation in fractional (i, j) index space.
    //
    // i_frac ∈ [0, ni-1],  j_frac ∈ [0, nj-1].
    //
    // For integer values of i_frac and j_frac (same-grid restart) this
    // reduces to an exact lookup with no floating-point blending.
    // ----------------------------------------------------------------
    [[nodiscard]] double Interp(const std::vector<double>& data,
                                 double i_frac, double j_frac) const noexcept
    {
        // Clamp to the valid point range.
        i_frac = std::clamp(i_frac, 0.0, static_cast<double>(ni - 1));
        j_frac = std::clamp(j_frac, 0.0, static_cast<double>(nj - 1));

        // Lower-left corner indices (clamped so i0+1 and j0+1 are valid).
        const int i0 = std::min(static_cast<int>(i_frac), ni - 2);
        const int j0 = std::min(static_cast<int>(j_frac), nj - 2);

        // Bilinear weights.
        const double ti = i_frac - static_cast<double>(i0);
        const double tj = j_frac - static_cast<double>(j0);

        // Four surrounding values.
        const double v00 = data[static_cast<std::size_t>( i0      +  j0      * ni)];
        const double v10 = data[static_cast<std::size_t>((i0 + 1) +  j0      * ni)];
        const double v01 = data[static_cast<std::size_t>( i0      + (j0 + 1) * ni)];
        const double v11 = data[static_cast<std::size_t>((i0 + 1) + (j0 + 1) * ni)];

        return (1.0 - ti) * (1.0 - tj) * v00
             +        ti  * (1.0 - tj) * v10
             + (1.0 - ti) *        tj  * v01
             +        ti  *        tj  * v11;
    }
};

// ============================================================
// Constructor / destructor
// ============================================================

VtkIC::VtkIC(const std::string& path)
    : impl_(std::make_unique<Impl>(path))
{}

VtkIC::~VtkIC() = default;

// ============================================================
// Apply
// ============================================================

void VtkIC::Apply(Fields& f, const Grid& grid,
                   const SimConfig& cfg, int l_start) const
{
    // ----------------------------------------------------------------
    // Fractional-index mapping from the new grid to the VTK grid.
    //
    // Axial   : i_frac = l_global_new × (ni_vtk - 1) / (L_max_new - 1)
    //           Maps new cell l_global ∈ [0, L_max_new-1] →
    //                 old cell range  [0, L_max_old-1] = [0, ni_vtk-2]
    //           For same grid (L_max_new == L_max_old):  i_frac == l_global.
    //
    // Radial  : j_frac = m_global_new × (nj_vtk - 1) / M_max_new
    //           Maps new cell m_global ∈ [0, M_max_new] →
    //                 old cell range  [0, M_max_old]   = [0, nj_vtk-1]
    //           For same grid (M_max_new == M_max_old):  j_frac == m_global.
    //
    // This preserves the fractional (z / L,  m_global / M_max) position of
    // each cell so restarts on a different resolution are well-defined.
    // ----------------------------------------------------------------

    // Denominators guarded against 1-cell degenerate grids.
    const double l_max_new = static_cast<double>(cfg.L_max);
    const double m_max_new = static_cast<double>(cfg.M_max);
    const double i_scale   = (cfg.L_max > 1)
                           ? static_cast<double>(impl_->ni - 1) / (l_max_new - 1.0)
                           : 0.0;
    const double j_scale   = (cfg.M_max > 0)
                           ? static_cast<double>(impl_->nj - 1) / m_max_new
                           : 0.0;

    const double gamma = cfg.gamma;

    for (int l = 1; l < f.rows - 1; ++l) {
        const int    l_global = l_start + l - 1;
        const double i_frac   = static_cast<double>(l_global) * i_scale;

        for (int m = 1; m < f.cols - 1; ++m) {
            // m_global: global radial index of this interior cell.
            // grid.m_start is the global m-index of local m = 1.
            const int    m_global = grid.m_start + m - 1;
            const double j_frac   = static_cast<double>(m_global) * j_scale;

            // ---- Interpolate all eight physical fields ----
            f.rho  [l][m] = impl_->Interp(impl_->rho,   i_frac, j_frac);
            f.v_z  [l][m] = impl_->Interp(impl_->v_z,   i_frac, j_frac);
            f.v_r  [l][m] = impl_->Interp(impl_->v_r,   i_frac, j_frac);
            f.v_phi[l][m] = impl_->Interp(impl_->v_phi, i_frac, j_frac);
            f.H_z  [l][m] = impl_->Interp(impl_->H_z,   i_frac, j_frac);
            f.H_r  [l][m] = impl_->Interp(impl_->H_r,   i_frac, j_frac);
            f.H_phi[l][m] = impl_->Interp(impl_->H_phi, i_frac, j_frac);
            f.e    [l][m] = impl_->Interp(impl_->e,     i_frac, j_frac);

            // ---- Derived scalars (same as ExpressionIC) ----
            f.p[l][m] = (gamma - 1.0) * f.rho[l][m] * f.e[l][m];
            f.P[l][m] = f.p[l][m]
                      + 0.5 * (f.H_z  [l][m] * f.H_z  [l][m]
                             + f.H_r  [l][m] * f.H_r  [l][m]
                             + f.H_phi[l][m] * f.H_phi[l][m]);
        }
    }
}
