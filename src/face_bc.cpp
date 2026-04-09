#include "face_bc.hpp"

#include <algorithm>

#include "bc_context.hpp"
#include "bcs/per_field_bc.hpp"
#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

// ============================================================
// Construction
// ============================================================

FaceBC::FaceBC(enum Face face, std::vector<BCSegment> segments)
    : face_(face), segments_(std::move(segments)) {}

// ============================================================
// Static factory
// ============================================================

auto FaceBC::FromConfig(enum Face face, const BCFaceConfig& face_cfg) -> FaceBC {
    std::vector<BCSegment> segs;
    segs.reserve(face_cfg.segments.size());

    for (const BCSegmentConfig& sc : face_cfg.segments) {
        BCSegment seg;
        seg.global_lo = sc.global_lo;
        seg.global_hi = sc.global_hi;
        seg.bc = std::make_unique<PerFieldBC>(face, sc);
        segs.push_back(std::move(seg));
    }

    return {face, std::move(segs)};
}

// ============================================================
// Helpers
// ============================================================

auto FaceBC::OwnsFace(const MPIManager& mpi) const noexcept -> bool {
    switch (face_) {
        case Face::L_LO:
            return mpi.IsLLoBoundary();
        case Face::L_HI:
            return mpi.IsLHiBoundary();
        case Face::M_LO:
            return mpi.IsMLoBoundary();
        case Face::M_HI:
            return mpi.IsMHiBoundary();
    }
    return false;
}

auto FaceBC::FaceGlobalRange(const SimConfig& cfg) const noexcept
    -> std::pair<int, int> {
    switch (face_) {
        case Face::L_LO:
        case Face::L_HI:
            return {0, cfg.M_max};
        case Face::M_LO:
        case Face::M_HI:
            return {0, cfg.L_max - 1};
    }
    return {0, 0};
}

auto FaceBC::ClipToLocal(int g_lo, int g_hi, const MPIManager& mpi, int& local_lo,
                           int& local_hi) const noexcept -> bool {
    int owned_global_lo, owned_global_hi;

    switch (face_) {
        case Face::L_LO:
        case Face::L_HI:
            owned_global_lo = mpi.m_start;
            owned_global_hi = mpi.m_end;
            break;
        case Face::M_LO:
        case Face::M_HI:
            owned_global_lo = mpi.l_start;
            owned_global_hi = mpi.l_end;
            break;
        default:
            return false;
    }

    const int clipped_lo = std::max(g_lo, owned_global_lo);
    const int clipped_hi = std::min(g_hi, owned_global_hi);

    if (clipped_lo > clipped_hi) {
        return false;
    }

    local_lo = clipped_lo - owned_global_lo + 1;
    local_hi = clipped_hi - owned_global_lo + 1;
    return true;
}

// ============================================================
// apply — main dispatch
// ============================================================

void FaceBC::Apply(Fields& f, const Grid& g, const SimConfig& cfg, const MPIManager& mpi,
                   double dt) const {
    if (!OwnsFace(mpi)) return;

    auto [face_lo, face_hi] = FaceGlobalRange(cfg);

    for (const BCSegment& seg : segments_) {
        const int g_lo = (seg.global_lo < 0) ? face_lo : seg.global_lo;
        const int g_hi = (seg.global_hi < 0) ? face_hi : seg.global_hi;

        int local_lo, local_hi;
        if (!ClipToLocal(g_lo, g_hi, mpi, local_lo, local_hi)) continue;

        // Corner policy for M_LO face (see Solver::advance() for ordering):
        // l=1 corners are owned by the L_LO BC on l-lo boundary ranks.
        // l=local_L corners are owned by the L_HI BC on l-hi boundary ranks.
        if (face_ == Face::M_LO) {
            if (mpi.IsLLoBoundary()) local_lo = std::max(local_lo, 2);
            if (mpi.IsLHiBoundary()) local_hi = std::min(local_hi, mpi.local_L - 1);
            if (local_lo > local_hi) continue;
        }

        BCContext ctx{.fields=f, .grid=g, .cfg=cfg, .mpi=mpi, .dt=dt, .local_lo=local_lo, .local_hi=local_hi};
        seg.bc->Apply(ctx);
    }
}
