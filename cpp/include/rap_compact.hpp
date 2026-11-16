#pragma once

#include "pit_types.hpp"
#include "simd_kernels.hpp"

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>

namespace pitmem {

enum class CompOp : uint8_t {
    LT = 0,
    LE = 1,
    GT = 2,
    GE = 3,
    EQ = 4,
    NE = 5,
    BETWEEN = 6
};

// Polarity / Typed-Anchor Constraint Flags
constexpr uint8_t FLAG_NEGATED       = 0x01; // Negative constraint ("shall not exceed / prohibited")
constexpr uint8_t FLAG_CROSS_DEFAULT = 0x02; // Triggers cross-default / acceleration
constexpr uint8_t FLAG_MAC_CLAUSE    = 0x04; // Material Adverse Change
constexpr uint8_t FLAG_CURE_RIGHT    = 0x08; // Equity cure right present

struct RiskAnchorInput {
    uint64_t source_mem_id = 0;
    std::string entity;
    std::string metric;
    std::string unit;
    std::string consequence;
    uint8_t op = 0;
    uint8_t flags = 0;
    double threshold_val = 0.0;
    double secondary_val = 0.0;
    uint32_t maturity_tick = 0;
    uint32_t tv_s = 0;
    uint32_t tv_e = INF_TICK;
    uint32_t tk_s = 0;
    uint32_t tk_e = INF_TICK;
};

#pragma pack(push, 1)
struct PackedAnchorTuple {
    uint64_t source_mem_id;
    uint16_t entity_id;
    uint16_t metric_id;
    uint16_t unit_id;
    uint16_t consequence_id;
    uint8_t  op;
    uint8_t  flags;
    uint16_t cluster_id;
    double   threshold_val;
    double   secondary_val;
    uint32_t maturity_tick;
    uint32_t tv_s;
    uint32_t tv_e;
    uint32_t tk_s;
    uint32_t tk_e;
}; // Exactly 56 bytes per lossless typed anchor tuple

struct PackedMemberSpan {
    uint64_t source_mem_id;
    uint16_t cluster_id;
    uint16_t snippet_len;
    uint32_t tv_s;
    uint32_t tv_e;
    uint32_t tk_s;
    uint32_t tk_e;
}; // 28 bytes per member span (64-bit source_mem_id + 4-tick PiT visibility guard)

struct SlottedPageHeader {
    uint32_t magic = 0x50495433u; // "PIT3"
    uint32_t num_clusters = 0;
    uint32_t num_anchors = 0;
    uint32_t num_members = 0;
    uint32_t dict_entries = 0;
    uint32_t sub_dim = 0;
    uint32_t dict_bytes = 0;
    uint32_t narrative_bytes = 0;
    uint32_t total_bytes = 0;
};
#pragma pack(pop)

struct CompactionStats {
    std::size_t raw_total_bytes = 0;
    std::size_t compacted_total_bytes = 0;
    double compression_ratio = 1.0;
    double space_fraction = 1.0;
    std::size_t num_input_records = 0;
    std::size_t num_clusters = 0;
    std::size_t num_preserved_anchors = 0;
};

struct HybridRAPResult {
    std::vector<uint16_t> cluster_ids;
    std::vector<uint64_t> member_ids;
    std::vector<float> cluster_distances;
    std::vector<std::string> cluster_summaries;
    std::vector<RiskAnchorInput> matched_anchors;
    double latency_us = 0.0;
};

class RAPCompactStore {
public:
    explicit RAPCompactStore(std::size_t orig_dim = 768) : orig_dim_(orig_dim) {}

    // Schema-generic cold-tier compaction:
    // Separates lossless 56-byte dictionary-coded Typed Anchor tuples from SQ8 residual narrative centroids.
    CompactionStats compact_batch(
        const std::vector<uint64_t>& mem_ids,
        const float* embeddings,
        const std::vector<std::string>& raw_texts,
        const std::vector<uint32_t>& tv_s,
        const std::vector<uint32_t>& tv_e,
        const std::vector<uint32_t>& tk_s,
        const std::vector<uint32_t>& tk_e,
        const std::vector<RiskAnchorInput>& anchors,
        double target_budget_ratio = 0.25) {
        clear();
        const std::size_t n = mem_ids.size();
        CompactionStats st;
        st.num_input_records = n;
        if (n == 0) return st;

        std::size_t raw_bytes = n * (orig_dim_ * sizeof(float) + 24);
        for (const auto& txt : raw_texts) raw_bytes += txt.size();
        st.raw_total_bytes = raw_bytes;

        double clamped = std::clamp(target_budget_ratio, 0.10, 0.65);
        std::size_t group_size = static_cast<std::size_t>(std::max(1.0, std::round(1.2 / clamped)));
        sub_dim_ = static_cast<std::size_t>(std::clamp<int>(
            static_cast<int>(std::round(orig_dim_ * clamped * 1.35 / 32.0)) * 32, 64, static_cast<int>(orig_dim_)));

        std::size_t target_clusters = std::min<std::size_t>(65535, (n + group_size - 1) / group_size);

        std::vector<uint16_t> rec_to_cluster(n, 0);
        std::vector<bool> assigned(n, false);
        cluster_centroids_sq8_.clear();
        cluster_metas_.clear();
        cluster_tv_s_.clear();
        cluster_tv_e_.clear();
        cluster_tk_s_.clear();
        cluster_tk_e_.clear();
        cluster_members_.clear();
        member_spans_.clear();
        member_snippets_.clear();
        member_spans_.reserve(n);
        member_snippets_.reserve(n);

        std::unordered_map<uint64_t, uint16_t> id_to_cluster;
        id_to_cluster.reserve(n);

        auto intervals_compatible = [&](std::size_t a, std::size_t b) noexcept -> bool {
            return (tv_s[a] < tv_e[b]) && (tv_s[b] < tv_e[a])
                && (tk_s[a] < tk_e[b]) && (tk_s[b] < tk_e[a]);
        };

        std::size_t cid = 0;
        std::vector<float> centroid_buf(orig_dim_, 0.0f);
        for (std::size_t seed = 0; seed < n && cid < 65535; ++seed) {
            if (assigned[seed]) continue;
            assigned[seed] = true;
            std::vector<std::size_t> members = {seed};

            const float* seed_vec = embeddings + seed * orig_dim_;
            if (group_size > 1 && cid + 1 < target_clusters) {
                std::vector<std::pair<float, std::size_t>> near_cands;
                std::size_t win_end = std::min<std::size_t>(n, seed + 256);
                for (std::size_t j = seed + 1; j < win_end; ++j) {
                    if (assigned[j] || !intervals_compatible(seed, j)) continue;
                    float d = simd::l2_sqr_fp32(seed_vec, embeddings + j * orig_dim_, orig_dim_);
                    near_cands.emplace_back(d, j);
                }
                std::sort(near_cands.begin(), near_cands.end());
                for (std::size_t m = 0; m < near_cands.size() && members.size() < group_size; ++m) {
                    std::size_t cand_j = near_cands[m].second;
                    bool all_compat = true;
                    for (std::size_t ex : members) {
                        if (!intervals_compatible(ex, cand_j)) { all_compat = false; break; }
                    }
                    if (all_compat) {
                        assigned[cand_j] = true;
                        members.push_back(cand_j);
                    }
                }
            } else {
                // Only merge mutually overlapping bi-temporal records; disjoint versions open a separate cluster
                for (std::size_t j = seed + 1; j < n && members.size() < group_size * 2; ++j) {
                    if (assigned[j] || !intervals_compatible(seed, j)) continue;
                    bool all_compat = true;
                    for (std::size_t ex : members) {
                        if (!intervals_compatible(ex, j)) { all_compat = false; break; }
                    }
                    if (all_compat) {
                        assigned[j] = true;
                        members.push_back(j);
                    }
                }
            }

            std::fill(centroid_buf.begin(), centroid_buf.end(), 0.0f);
            uint32_t c_tvs = INF_TICK, c_tve = 0, c_tks = INF_TICK, c_tke = 0;
            std::vector<uint32_t> m_indices;
            m_indices.reserve(members.size());

            for (std::size_t idx : members) {
                rec_to_cluster[idx] = static_cast<uint16_t>(cid);
                id_to_cluster[mem_ids[idx]] = static_cast<uint16_t>(cid);
                const float* mv = embeddings + idx * orig_dim_;
                for (std::size_t d = 0; d < orig_dim_; ++d) centroid_buf[d] += mv[d];
                if (tv_s[idx] < c_tvs) c_tvs = tv_s[idx];
                if (tv_e[idx] > c_tve) c_tve = tv_e[idx];
                if (tk_s[idx] < c_tks) c_tks = tk_s[idx];
                if (tk_e[idx] > c_tke) c_tke = tk_e[idx];

                std::size_t per_mem_chars = static_cast<std::size_t>(
                    std::max<double>(24.0, (raw_texts[idx].size() * clamped * 0.65) / static_cast<double>(members.size())));
                std::string snip = raw_texts[idx].substr(0, std::min(raw_texts[idx].size(), per_mem_chars));
                PackedMemberSpan ms{};
                ms.source_mem_id = mem_ids[idx];
                ms.cluster_id = static_cast<uint16_t>(cid);
                ms.snippet_len = static_cast<uint16_t>(snip.size());
                ms.tv_s = tv_s[idx];
                ms.tv_e = tv_e[idx];
                ms.tk_s = tk_s[idx];
                ms.tk_e = tk_e[idx];
                m_indices.push_back(static_cast<uint32_t>(member_spans_.size()));
                member_spans_.push_back(ms);
                member_snippets_.push_back(std::move(snip));
            }
            float inv_m = 1.0f / static_cast<float>(members.size());
            for (std::size_t d = 0; d < orig_dim_; ++d) centroid_buf[d] *= inv_m;

            // L2-normalize the sub_dim_ prefix onto the unit hypersphere so cold distances
            // match hot BiT-HNSW squared-L2 distances: ||q_sub - c_sub||_2^2 = 2 - 2 cos_sub(q, c) in [0, 4]
            double sub_norm_sq = 0.0;
            for (std::size_t d = 0; d < sub_dim_; ++d) {
                sub_norm_sq += static_cast<double>(centroid_buf[d]) * static_cast<double>(centroid_buf[d]);
            }
            if (sub_norm_sq > 1e-18) {
                float inv_sub_norm = static_cast<float>(1.0 / std::sqrt(sub_norm_sq));
                for (std::size_t d = 0; d < sub_dim_; ++d) centroid_buf[d] *= inv_sub_norm;
            }

            cluster_centroids_sq8_.resize((cid + 1) * sub_dim_, 0);
            cluster_metas_.push_back(simd::quantize_db_sq8(
                centroid_buf.data(),
                cluster_centroids_sq8_.data() + cid * sub_dim_,
                sub_dim_));
            cluster_tv_s_.push_back(c_tvs);
            cluster_tv_e_.push_back(c_tve);
            cluster_tk_s_.push_back(c_tks);
            cluster_tk_e_.push_back(c_tke);
            cluster_members_.push_back(std::move(m_indices));
            cid++;
        }
        num_clusters_ = cid;
        st.num_clusters = num_clusters_;

        packed_anchors_.reserve(anchors.size());
        for (const auto& a : anchors) {
            PackedAnchorTuple pt{};
            pt.source_mem_id = a.source_mem_id;
            pt.entity_id = intern_string(a.entity);
            pt.metric_id = intern_string(a.metric);
            pt.unit_id = intern_string(a.unit);
            pt.consequence_id = intern_string(a.consequence);
            pt.op = a.op;
            pt.flags = a.flags;
            auto cit = id_to_cluster.find(a.source_mem_id);
            pt.cluster_id = (cit != id_to_cluster.end()) ? cit->second : 0;
            pt.threshold_val = a.threshold_val;
            pt.secondary_val = a.secondary_val;
            pt.maturity_tick = a.maturity_tick;
            pt.tv_s = a.tv_s;
            pt.tv_e = a.tv_e;
            pt.tk_s = a.tk_s;
            pt.tk_e = a.tk_e;
            packed_anchors_.push_back(pt);
        }
        st.num_preserved_anchors = packed_anchors_.size();

        serialize_slotted_page();
        packed_anchors_.clear();
        id_to_str_.clear();
        str_to_id_.clear();
        deserialize_slotted_page(page_blob_.data(), page_blob_.size());
        st.compacted_total_bytes = page_blob_.size();
        st.compression_ratio = (st.compacted_total_bytes > 0)
            ? static_cast<double>(st.raw_total_bytes) / static_cast<double>(st.compacted_total_bytes)
            : 1.0;
        st.space_fraction = (st.raw_total_bytes > 0)
            ? static_cast<double>(st.compacted_total_bytes) / static_cast<double>(st.raw_total_bytes)
            : 1.0;
        return st;
    }

    bool find_member_bounds(
        uint64_t mem_id,
        uint32_t& out_tvs,
        uint32_t& out_tve,
        uint32_t& out_tks,
        uint32_t& out_tke) const noexcept {
        for (const auto& ms : member_spans_) {
            if (ms.source_mem_id == mem_id) {
                out_tvs = ms.tv_s;
                out_tve = ms.tv_e;
                out_tks = ms.tk_s;
                out_tke = ms.tk_e;
                return true;
            }
        }
        return false;
    }

    // Point-in-Time Anchor Lookup (with inverted entity posting index + optional post-compaction retraction overlay)
    std::vector<RiskAnchorInput> query_anchors_pit(
        const std::string& entity_filter,
        uint32_t tv_q,
        uint32_t tk_q,
        const std::unordered_map<uint64_t, uint32_t>* retracted_tk = nullptr) const {
        auto anchor_vis = [&](const PackedAnchorTuple& pt) noexcept -> bool {
            uint32_t eff_tke = pt.tk_e;
            if (retracted_tk != nullptr) {
                auto rit = retracted_tk->find(pt.source_mem_id);
                if (rit != retracted_tk->end() && rit->second < eff_tke) eff_tke = rit->second;
            }
            return bitemp_visible(pt.tv_s, pt.tv_e, pt.tk_s, eff_tke, tv_q, tk_q);
        };
        std::vector<RiskAnchorInput> hits;
        if (!entity_filter.empty()) {
            auto it = str_to_id_.find(entity_filter);
            if (it == str_to_id_.end()) return hits;
            auto pit = entity_postings_.find(it->second);
            if (pit == entity_postings_.end()) return hits;
            for (uint32_t a_idx : pit->second) {
                const auto& pt = packed_anchors_[a_idx];
                if (!anchor_vis(pt)) continue;
                hits.push_back(unpack_anchor(pt));
            }
            return hits;
        }

        for (const auto& pt : packed_anchors_) {
            if (!anchor_vis(pt)) continue;
            hits.push_back(unpack_anchor(pt));
        }
        return hits;
    }

    // Unified Cold-Tier Hybrid Search (co-executing AVX2 SQ8 Residual Centroid Search + Lossless Bi-Temporal Anchor Join)
    HybridRAPResult search_hybrid_pit(
        const float* q_vec,
        const std::string& entity_filter,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t top_k = 5,
        const std::unordered_map<uint64_t, uint32_t>* retracted_tk = nullptr) const {
        auto t0 = std::chrono::steady_clock::now();
        HybridRAPResult out;
        if (num_clusters_ == 0 && packed_anchors_.empty()) return out;

        // Unit-normalize query prefix onto S^{sub_dim_-1} so cold distances align with hot BiT-HNSW [0, 4] scale
        float q_sub_stack[1024];
        std::vector<float> q_sub_heap;
        float* q_sub = q_sub_stack;
        if (sub_dim_ > 1024) {
            q_sub_heap.resize(sub_dim_);
            q_sub = q_sub_heap.data();
        }
        double q_norm_sq = 0.0;
        for (std::size_t d = 0; d < sub_dim_; ++d) {
            q_sub[d] = q_vec[d];
            q_norm_sq += static_cast<double>(q_vec[d]) * static_cast<double>(q_vec[d]);
        }
        if (q_norm_sq > 1e-18) {
            float inv_qn = static_cast<float>(1.0 / std::sqrt(q_norm_sq));
            for (std::size_t d = 0; d < sub_dim_; ++d) q_sub[d] *= inv_qn;
        }
        simd::SQ8Query sq_q = simd::prepare_sq8_query(q_sub, sub_dim_);
        struct CandEntry {
            float dist;
            uint16_t cid;
            uint32_t vis_member_idx;
            bool operator<(const CandEntry& o) const noexcept {
                return (dist != o.dist) ? (dist < o.dist) : (cid < o.cid);
            }
        };
        std::vector<CandEntry> cands;
        cands.reserve(num_clusters_);

        for (std::size_t c = 0; c < num_clusters_; ++c) {
            // Coarse cluster bounding-box pruning (zero false negatives)
            if (!bitemp_visible(cluster_tv_s_[c], cluster_tv_e_[c], cluster_tk_s_[c], cluster_tk_e_[c], tv_q, tk_q)) {
                continue;
            }
            // Exact member-level bi-temporal guard (eliminates bounding-box false positives & look-ahead/stale snippets!)
            uint32_t first_vis_mi = UINT32_MAX;
            if (c < cluster_members_.size()) {
                for (uint32_t mi : cluster_members_[c]) {
                    const auto& ms = member_spans_[mi];
                    uint32_t eff_tke = ms.tk_e;
                    if (retracted_tk != nullptr) {
                        auto rit = retracted_tk->find(ms.source_mem_id);
                        if (rit != retracted_tk->end() && rit->second < eff_tke) eff_tke = rit->second;
                    }
                    if (bitemp_visible(ms.tv_s, ms.tv_e, ms.tk_s, eff_tke, tv_q, tk_q)) {
                        first_vis_mi = mi;
                        break;
                    }
                }
            }
            if (first_vis_mi == UINT32_MAX) continue;

            float d = simd::sq8_approx_l2_sqr(
                sq_q, cluster_centroids_sq8_.data() + c * sub_dim_, cluster_metas_[c]);
            cands.push_back(CandEntry{d, static_cast<uint16_t>(c), first_vis_mi});
        }

        std::size_t keep = std::min<std::size_t>(top_k, cands.size());
        if (keep > 0) {
            std::partial_sort(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(keep), cands.end());
        }

        std::unordered_set<uint16_t> top_cset;
        for (std::size_t i = 0; i < keep; ++i) {
            uint16_t cid = cands[i].cid;
            uint32_t mi = cands[i].vis_member_idx;
            out.cluster_ids.push_back(cid);
            out.member_ids.push_back(member_spans_[mi].source_mem_id);
            out.cluster_distances.push_back(cands[i].dist);
            out.cluster_summaries.push_back(member_snippets_[mi]);
            top_cset.insert(cid);
        }

        // Join bi-temporally visible lossless anchors (respecting post-compaction retractions)
        if (!entity_filter.empty()) {
            out.matched_anchors = query_anchors_pit(entity_filter, tv_q, tk_q, retracted_tk);
        } else {
            for (const auto& pt : packed_anchors_) {
                uint32_t eff_tke = pt.tk_e;
                if (retracted_tk != nullptr) {
                    auto rit = retracted_tk->find(pt.source_mem_id);
                    if (rit != retracted_tk->end() && rit->second < eff_tke) eff_tke = rit->second;
                }
                if (top_cset.count(pt.cluster_id) && bitemp_visible(pt.tv_s, pt.tv_e, pt.tk_s, eff_tke, tv_q, tk_q)) {
                    out.matched_anchors.push_back(unpack_anchor(pt));
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        out.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }

    void save_segment_file(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs) throw std::runtime_error("Cannot open RAP segment file for write: " + path);
        ofs.write(reinterpret_cast<const char*>(page_blob_.data()), static_cast<std::streamsize>(page_blob_.size()));
    }

    void load_segment_file(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs) throw std::runtime_error("Cannot open RAP segment file for read: " + path);
        std::size_t sz = static_cast<std::size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        page_blob_.resize(sz);
        if (sz > 0) ifs.read(reinterpret_cast<char*>(page_blob_.data()), static_cast<std::streamsize>(sz));
        deserialize_slotted_page(page_blob_.data(), page_blob_.size());
    }

    std::size_t serialized_bytes() const noexcept { return page_blob_.size(); }

    void clear() {
        str_to_id_.clear();
        id_to_str_.clear();
        packed_anchors_.clear();
        entity_postings_.clear();
        cluster_centroids_sq8_.clear();
        cluster_metas_.clear();
        cluster_tv_s_.clear();
        cluster_tv_e_.clear();
        cluster_tk_s_.clear();
        cluster_tk_e_.clear();
        cluster_members_.clear();
        member_spans_.clear();
        member_snippets_.clear();
        page_blob_.clear();
        num_clusters_ = 0;
    }

private:
    std::size_t orig_dim_ = 768;
    std::size_t sub_dim_ = 256;
    std::size_t num_clusters_ = 0;

    std::unordered_map<std::string, uint16_t> str_to_id_;
    std::vector<std::string> id_to_str_;
    std::vector<PackedAnchorTuple> packed_anchors_;
    std::unordered_map<uint16_t, std::vector<uint32_t>> entity_postings_;

    std::vector<int8_t> cluster_centroids_sq8_;
    std::vector<simd::SQ8VectorMeta> cluster_metas_;
    std::vector<uint32_t> cluster_tv_s_, cluster_tv_e_, cluster_tk_s_, cluster_tk_e_;
    std::vector<std::vector<uint32_t>> cluster_members_;
    std::vector<PackedMemberSpan> member_spans_;
    std::vector<std::string> member_snippets_;

    std::vector<uint8_t> page_blob_;

    RiskAnchorInput unpack_anchor(const PackedAnchorTuple& pt) const {
        RiskAnchorInput out;
        out.source_mem_id = pt.source_mem_id;
        out.entity = id_to_str_[pt.entity_id];
        out.metric = id_to_str_[pt.metric_id];
        out.unit = id_to_str_[pt.unit_id];
        out.consequence = id_to_str_[pt.consequence_id];
        out.op = pt.op;
        out.flags = pt.flags;
        out.threshold_val = pt.threshold_val;
        out.secondary_val = pt.secondary_val;
        out.maturity_tick = pt.maturity_tick;
        out.tv_s = pt.tv_s;
        out.tv_e = pt.tv_e;
        out.tk_s = pt.tk_s;
        out.tk_e = pt.tk_e;
        return out;
    }

    uint16_t intern_string(const std::string& s) {
        auto it = str_to_id_.find(s);
        if (it != str_to_id_.end()) return it->second;
        uint16_t id = static_cast<uint16_t>(id_to_str_.size());
        id_to_str_.push_back(s);
        str_to_id_[s] = id;
        return id;
    }

    void serialize_slotted_page() {
        std::size_t dict_bytes = 0;
        for (const auto& s : id_to_str_) dict_bytes += sizeof(uint16_t) + s.size();

        std::size_t nar_bytes = member_spans_.size() * sizeof(PackedMemberSpan);
        for (const auto& snip : member_snippets_) {
            nar_bytes += snip.size();
        }

        std::size_t centroid_bytes = num_clusters_ * (
            sub_dim_ * sizeof(int8_t) + sizeof(simd::SQ8VectorMeta) + 4 * sizeof(uint32_t));
        std::size_t anchor_bytes = packed_anchors_.size() * sizeof(PackedAnchorTuple);
        std::size_t total = sizeof(SlottedPageHeader) + dict_bytes + anchor_bytes + centroid_bytes + nar_bytes;

        page_blob_.assign(total, 0);
        uint8_t* ptr = page_blob_.data();

        SlottedPageHeader hdr{};
        hdr.num_clusters = static_cast<uint32_t>(num_clusters_);
        hdr.num_anchors = static_cast<uint32_t>(packed_anchors_.size());
        hdr.num_members = static_cast<uint32_t>(member_spans_.size());
        hdr.dict_entries = static_cast<uint32_t>(id_to_str_.size());
        hdr.sub_dim = static_cast<uint32_t>(sub_dim_);
        hdr.dict_bytes = static_cast<uint32_t>(dict_bytes);
        hdr.narrative_bytes = static_cast<uint32_t>(nar_bytes);
        hdr.total_bytes = static_cast<uint32_t>(total);
        std::memcpy(ptr, &hdr, sizeof(hdr));
        ptr += sizeof(hdr);

        for (const auto& s : id_to_str_) {
            uint16_t len = static_cast<uint16_t>(s.size());
            std::memcpy(ptr, &len, sizeof(len));
            ptr += sizeof(len);
            if (len > 0) std::memcpy(ptr, s.data(), len);
            ptr += len;
        }
        if (anchor_bytes > 0) {
            std::memcpy(ptr, packed_anchors_.data(), anchor_bytes);
            ptr += anchor_bytes;
        }
        if (num_clusters_ > 0) {
            std::memcpy(ptr, cluster_centroids_sq8_.data(), num_clusters_ * sub_dim_);
            ptr += num_clusters_ * sub_dim_;
            std::memcpy(ptr, cluster_metas_.data(), num_clusters_ * sizeof(simd::SQ8VectorMeta));
            ptr += num_clusters_ * sizeof(simd::SQ8VectorMeta);
            for (std::size_t c = 0; c < num_clusters_; ++c) {
                uint32_t ts[4] = {cluster_tv_s_[c], cluster_tv_e_[c], cluster_tk_s_[c], cluster_tk_e_[c]};
                std::memcpy(ptr, ts, sizeof(ts));
                ptr += sizeof(ts);
            }
        }
        if (!member_spans_.empty()) {
            std::size_t mspan_bytes = member_spans_.size() * sizeof(PackedMemberSpan);
            std::memcpy(ptr, member_spans_.data(), mspan_bytes);
            ptr += mspan_bytes;
            for (const auto& snip : member_snippets_) {
                if (!snip.empty()) {
                    std::memcpy(ptr, snip.data(), snip.size());
                    ptr += snip.size();
                }
            }
        }
    }

    void deserialize_slotted_page(const uint8_t* data, std::size_t len) {
        if (len < sizeof(SlottedPageHeader)) return;
        SlottedPageHeader hdr{};
        std::memcpy(&hdr, data, sizeof(hdr));
        if (hdr.magic != 0x50495433u || hdr.total_bytes > len) return;

        const uint8_t* ptr = data + sizeof(SlottedPageHeader);
        str_to_id_.clear();
        id_to_str_.assign(hdr.dict_entries, std::string());
        for (uint32_t i = 0; i < hdr.dict_entries; ++i) {
            uint16_t slen = 0;
            std::memcpy(&slen, ptr, sizeof(slen));
            ptr += sizeof(slen);
            id_to_str_[i].assign(reinterpret_cast<const char*>(ptr), slen);
            str_to_id_[id_to_str_[i]] = static_cast<uint16_t>(i);
            ptr += slen;
        }

        packed_anchors_.resize(hdr.num_anchors);
        entity_postings_.clear();
        if (hdr.num_anchors > 0) {
            std::memcpy(packed_anchors_.data(), ptr, hdr.num_anchors * sizeof(PackedAnchorTuple));
            ptr += hdr.num_anchors * sizeof(PackedAnchorTuple);
            for (uint32_t ai = 0; ai < hdr.num_anchors; ++ai) {
                entity_postings_[packed_anchors_[ai].entity_id].push_back(ai);
            }
        }

        num_clusters_ = hdr.num_clusters;
        sub_dim_ = hdr.sub_dim;
        cluster_members_.assign(num_clusters_, {});
        if (num_clusters_ > 0) {
            cluster_centroids_sq8_.resize(num_clusters_ * sub_dim_);
            std::memcpy(cluster_centroids_sq8_.data(), ptr, num_clusters_ * sub_dim_);
            ptr += num_clusters_ * sub_dim_;
            cluster_metas_.resize(num_clusters_);
            std::memcpy(cluster_metas_.data(), ptr, num_clusters_ * sizeof(simd::SQ8VectorMeta));
            ptr += num_clusters_ * sizeof(simd::SQ8VectorMeta);
            cluster_tv_s_.resize(num_clusters_);
            cluster_tv_e_.resize(num_clusters_);
            cluster_tk_s_.resize(num_clusters_);
            cluster_tk_e_.resize(num_clusters_);
            for (std::size_t c = 0; c < num_clusters_; ++c) {
                uint32_t ts[4];
                std::memcpy(ts, ptr, sizeof(ts));
                ptr += sizeof(ts);
                cluster_tv_s_[c] = ts[0];
                cluster_tv_e_[c] = ts[1];
                cluster_tk_s_[c] = ts[2];
                cluster_tk_e_[c] = ts[3];
            }
        }

        member_spans_.resize(hdr.num_members);
        member_snippets_.assign(hdr.num_members, std::string());
        if (hdr.num_members > 0) {
            std::size_t mspan_bytes = hdr.num_members * sizeof(PackedMemberSpan);
            std::memcpy(member_spans_.data(), ptr, mspan_bytes);
            ptr += mspan_bytes;
            for (uint32_t mi = 0; mi < hdr.num_members; ++mi) {
                uint16_t slen = member_spans_[mi].snippet_len;
                if (slen > 0) {
                    member_snippets_[mi].assign(reinterpret_cast<const char*>(ptr), slen);
                    ptr += slen;
                }
                if (member_spans_[mi].cluster_id < num_clusters_) {
                    cluster_members_[member_spans_[mi].cluster_id].push_back(mi);
                }
            }
        }
    }
};

} // namespace pitmem
