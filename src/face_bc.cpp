#include "face_bc.hpp"

#include <algorithm>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "bc_context.hpp"
#include "bc_registry.hpp"
#include "config.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "mpi_manager.hpp"

// ============================================================
// Construction
// ============================================================

FaceBC::FaceBC(Face face, std::vector<BCSegment> segments)
    : face_(face), segments_(std::move(segments))
{}

// ============================================================
// Static factory
// ============================================================

FaceBC FaceBC::from_config(Face face, const BCFaceConfig& face_cfg) {
    BCRegistry& reg = BCRegistry::instance();
    std::vector<BCSegment> segs;
    segs.reserve(face_cfg.segments.size());

    for (const BCSegmentConfig& sc : face_cfg.segments) {
        BCSegment seg;
        seg.global_lo = sc.global_lo;
        seg.global_hi = sc.global_hi;

        // Parse the per-segment params YAML (may be an empty string → null node).
        YAML::Node params;
        if (!sc.params_yaml.empty())
            params = YAML::Load(sc.params_yaml);

        seg.bc = reg.create(sc.type, params);
        segs.push_back(std::move(seg));
    }

    return FaceBC(face, std::move(segs));
}

// ============================================================
// Helpers
// ============================================================

bool FaceBC::owns_face(const MPIManager& mpi) const noexcept {
    switch (face_) {
        case Face::L_LO: return mpi.is_l_lo_boundary();
        case Face::L_HI: return mpi.is_l_hi_boundary();
        case Face::M_LO: return mpi.is_m_lo_boundary();
        case Face::M_HI: return mpi.is_m_hi_boundary();
    }
    return false;
}

// Returns the full [lo, hi] range (inclusive) of the face's *free* axis in
// global coordinates.
//
//   L_LO / L_HI : free axis = m,  range = [0, M_max]
//   M_LO / M_HI : free axis = l,  range = [0, L_max_global - 1]
std::pair<int,int>
FaceBC::face_global_range(const SimConfig& cfg) const noexcept {
    switch (face_) {
        case Face::L_LO:
        case Face::L_HI:
            return { 0, cfg.M_max };
        case Face::M_LO:
        case Face::M_HI:
            return { 0, cfg.L_max_global - 1 };
    }
    return { 0, 0 };
}

// Clip the global segment [g_lo, g_hi] to what this rank owns, and convert
// the result to local (1-based interior) indices.
//
// For L faces the rank owns m ∈ [m_start, m_end] → local m ∈ [1, local_M].
// For M faces the rank owns l ∈ [l_start, l_end] → local l ∈ [1, local_L].
//
// Returns false when the intersection is empty (segment does not touch this rank).
bool FaceBC::clip_to_local(int g_lo, int g_hi, const MPIManager& mpi,
                            int& local_lo, int& local_hi) const noexcept {
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

    if (clipped_lo > clipped_hi)
        return false;   // empty intersection — this rank owns no cells in this segment

    // Convert from global to local 1-based interior index:
    //   local = global - owned_start + 1
    local_lo = clipped_lo - owned_global_lo + 1;
    local_hi = clipped_hi - owned_global_lo + 1;
    return true;
}

// ============================================================
// apply — main dispatch
// ============================================================

void FaceBC::apply(Fields& f, const Grid& g, const SimConfig& cfg,
                   const MPIManager& mpi, double dt) const {
    if (!owns_face(mpi))
        return;

    auto [face_lo, face_hi] = face_global_range(cfg);

    for (const BCSegment& seg : segments_) {
        // Resolve negative sentinels to the real face bounds.
        const int g_lo = (seg.global_lo < 0) ? face_lo : seg.global_lo;
        const int g_hi = (seg.global_hi < 0) ? face_hi : seg.global_hi;

        int local_lo, local_hi;
        if (!clip_to_local(g_lo, g_hi, mpi, local_lo, local_hi))
            continue;   // this rank owns no cells from this segment

        // Corner policy for M_LO face:
        //
        // The L_LO face (InflowBC) owns the corner cell (l=1, m=1) and sets
        // it before any M_LO BC runs.  To avoid overwriting that, M_LO
        // segments must not touch l=1 on l-lo boundary ranks.  We enforce
        // this by clamping local_lo upward on those ranks.
        //
        // Similarly, the L_HI face owns (l=local_L, m=1) on the l-hi boundary.
        if (face_ == Face::M_LO) {
            if (mpi.is_l_lo_boundary())
                local_lo = std::max(local_lo, 2);
            if (mpi.is_l_hi_boundary())
                local_hi = std::min(local_hi, mpi.local_L - 1);
            if (local_lo > local_hi)
                continue;
        }

        BCContext ctx{ f, g, cfg, mpi, dt, local_lo, local_hi };
        seg.bc->apply(ctx);
    }
}
