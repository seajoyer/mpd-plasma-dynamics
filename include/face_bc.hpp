#pragma once

#include <memory>
#include <utility>
#include <vector>
#include "iboundary_condition.hpp"

struct BCFaceConfig;
struct BCSegmentConfig;
struct SimConfig;
class Fields;
class Grid;
class MPIManager;

/// One contiguous segment of a face covered by a single IBoundaryCondition.
///
/// Global-index convention (free axis)
/// ─────────────────────────────────────
///   global_lo < 0  →  treated as the first index of the face (0)
///   global_hi < 0  →  treated as the last  index of the face
struct BCSegment {
    int global_lo{-1};  ///< negative → face start
    int global_hi{-1};  ///< negative → face end
    std::unique_ptr<IBoundaryCondition> bc;
};

/// Manages one face of the Cartesian domain as an ordered list of non-overlapping
/// segments, each handled by its own IBoundaryCondition.
///
/// Responsibilities
/// ─────────────────
///   1. Ownership check  — skips the whole face if this rank does not own it.
///   2. Range resolution — converts negative sentinel values to real indices.
///   3. Domain clipping  — intersects each segment's global range with the
///                         subset of the face that this rank owns, converting
///                         the result to local indices passed via BCContext.
///   4. Corner policy    — for the M_LO face, l=1 corners are excluded on l-lo
///                         boundary ranks and l=local_L corners are excluded on
///                         l-hi boundary ranks; this matches the BC application
///                         order in Solver::advance().
///
/// Construction
/// ─────────────
/// Build a FaceBC from config with the static factory:
///
///   auto bc = FaceBC::from_config(FaceBC::Face::M_LO, cfg.bc_m_lo);
///
/// This creates a PerFieldBC for each segment directly from BCSegmentConfig,
/// with no external registry lookup required.
class FaceBC {
public:
    enum class Face { L_LO, L_HI, M_LO, M_HI };

    FaceBC(Face face, std::vector<BCSegment> segments);

    // Non-copyable, movable.
    FaceBC(const FaceBC&)            = delete;
    auto operator=(const FaceBC&) -> FaceBC& = delete;
    FaceBC(FaceBC&&)                 = default;
    auto operator=(FaceBC&&) -> FaceBC&      = default;

    /// Build from config, creating a PerFieldBC for each segment.
    static auto FromConfig(Face face, const BCFaceConfig& face_cfg) -> FaceBC;

    /// Apply all segments to the appropriate cells on this rank.
    /// No-op if this rank does not own the face.
    void Apply(Fields& f, const Grid& g, const SimConfig& cfg,
               const MPIManager& mpi, double dt) const;

    [[nodiscard]] auto Face() const -> Face { return face_; }

private:
    enum Face                   face_;
    std::vector<BCSegment> segments_;

    [[nodiscard]] auto OwnsFace(const MPIManager& mpi) const noexcept -> bool;

    /// Real (non-sentinel) global range of the face's free axis.
    [[nodiscard]] auto FaceGlobalRange(const SimConfig& cfg) const noexcept -> std::pair<int,int>;

    /// Clip [g_lo, g_hi] to this rank's owned slice and convert to local indices.
    /// Returns false if the intersection is empty.
    auto ClipToLocal(int g_lo, int g_hi, const MPIManager& mpi,
                       int& local_lo, int& local_hi) const noexcept -> bool;
};
