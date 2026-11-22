#pragma once

#include "pit_types.hpp"
#include "simd_kernels.hpp"
#include "hnswlib/hnswlib.h"

#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pitmem {

// Official Upstream Open-Source Library Baseline: nmslib/hnswlib v0.8.0 (Malkov & Yashunin)
// Multi-layer HierarchicalNSW<float> + AVX2 L2Space + native C++ BaseFilterFunctor for 4-sided Bi-Temporal Filtering!
class OfficialHNSWLibIndex {
public:
    OfficialHNSWLibIndex(std::size_t dim, uint32_t m = 16, uint32_t ef_construction = 96)
        : dim_(dim), m_(m), ef_c_(ef_construction), space_(dim) {}

    void build_batch(
        const uint64_t* ext_ids,
        const float* vectors,
        const uint32_t* tv_s,
        const uint32_t* tv_e,
        const uint32_t* tk_s,
        const uint32_t* tk_e,
        std::size_t n,
        int num_threads = 0) {
        if (n == 0) return;
        std::size_t base = n_;
        n_ += n;
        if (!hnsw_) {
            hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space_, std::max<std::size_t>(n_ + 1024, 2048), m_, ef_c_);
        } else if (n_ > hnsw_->max_elements_) {
            hnsw_->resizeIndex(std::max<std::size_t>(n_ + 1024, hnsw_->max_elements_ * 2));
        }

        ext_ids_.resize(n_);
        tv_s_.resize(n_);
        tv_e_.resize(n_);
        tk_s_.resize(n_);
        tk_e_.resize(n_);
        ext_to_int_.reserve(n_);

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t id = base + i;
            ext_ids_[id] = ext_ids[i];
            ext_to_int_[ext_ids[i]] = static_cast<uint32_t>(id);
            tv_s_[id] = tv_s[i];
            tv_e_[id] = tv_e[i];
            tk_s_[id] = tk_s[i];
            tk_e_[id] = tk_e[i];
        }

        if (base == 0) {
            hnsw_->addPoint(vectors, static_cast<hnswlib::labeltype>(0));
        }
        std::size_t start_i = (base == 0) ? 1 : 0;

        #ifdef _OPENMP
        int prev_t = omp_get_max_threads();
        if (num_threads > 0) omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 32) if(n > 64)
        #endif
        for (std::size_t i = start_i; i < n; ++i) {
            std::size_t id = base + i;
            hnsw_->addPoint(vectors + i * dim_, static_cast<hnswlib::labeltype>(id));
        }
        #ifdef _OPENMP
        if (num_threads > 0) omp_set_num_threads(prev_t);
        #endif
    }

    void update_tx_end(uint64_t ext_id, uint32_t new_tk_e) {
        auto it = ext_to_int_.find(ext_id);
        if (it != ext_to_int_.end()) tk_e_[it->second] = new_tk_e;
    }

    QueryResult search(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t ef_search = 64) const {
        auto t0 = std::chrono::steady_clock::now();
        QueryResult out;
        if (!hnsw_ || n_ == 0 || k == 0) return out;

        struct BiTempFunctor final : public hnswlib::BaseFilterFunctor {
            const uint32_t* tv_s;
            const uint32_t* tv_e;
            const uint32_t* tk_s;
            const uint32_t* tk_e;
            uint32_t tv_q;
            uint32_t tk_q;
            uint64_t* checks;
            bool operator()(hnswlib::labeltype id) override {
                (*checks)++;
                return bitemp_visible(tv_s[id], tv_e[id], tk_s[id], tk_e[id], tv_q, tk_q);
            }
        };

        hnsw_->setEf(std::max<std::size_t>(k, ef_search));
        uint64_t pred_cnt = 0;
        BiTempFunctor filt;
        filt.tv_s = tv_s_.data;
        filt.tv_e = tv_e_.data;
        filt.tk_s = tk_s_.data;
        filt.tk_e = tk_e_.data;
        filt.tv_q = tv_q;
        filt.tk_q = tk_q;
        filt.checks = &pred_cnt;

        auto res = hnsw_->searchKnnCloserFirst(q_vec, k, &filt);
        out.stats.predicate_checks = pred_cnt;
        out.ids.resize(res.size());
        out.distances.resize(res.size());
        for (std::size_t i = 0; i < res.size(); ++i) {
            out.distances[i] = res[i].first;
            out.ids[i] = ext_ids_[res[i].second];
        }
        auto t1 = std::chrono::steady_clock::now();
        out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }

    // Exact multi-layer ACORN-1 arbitrary-predicate 2-hop search (Patel et al., SIGMOD 2024)
    // executed directly over the genuine upstream hnswlib::HierarchicalNSW<float> multi-layer hierarchy!
    QueryResult search_acorn1(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t ef_search = 64) const {
        auto t0 = std::chrono::steady_clock::now();
        QueryResult out;
        if (!hnsw_ || n_ == 0 || k == 0 || hnsw_->cur_element_count == 0) return out;

        const std::size_t cur_cnt = hnsw_->cur_element_count;
        auto passes = [&](hnswlib::tableint iid) -> bool {
            out.stats.predicate_checks++;
            std::size_t lbl = static_cast<std::size_t>(hnsw_->getExternalLabel(iid));
            return bitemp_visible(tv_s_[lbl], tv_e_[lbl], tk_s_[lbl], tk_e_[lbl], tv_q, tk_q);
        };
        auto dist_to_q = [&](hnswlib::tableint iid) -> float {
            out.stats.distance_computations++;
            return hnsw_->fstdistfunc_(q_vec, hnsw_->getDataByInternalId(iid), hnsw_->dist_func_param_);
        };

        // Stage 1: Multi-layer greedy 2-hop predicate-aware zoom-in (matching ACORN.cpp greedy_update_nearest)
        hnswlib::tableint currObj = hnsw_->enterpoint_node_;
        bool curr_pass = passes(currObj);
        float curdist = dist_to_q(currObj);

        for (int level = hnsw_->maxlevel_; level > 0; --level) {
            while (true) {
                hnswlib::tableint prev_nearest = currObj;
                hnswlib::linklistsizeint* ll_cur = hnsw_->get_linklist(currObj, level);
                unsigned short sz = hnsw_->getListCount(ll_cur);
                auto* data = reinterpret_cast<hnswlib::tableint*>(ll_cur + 1);
                uint32_t num_found = 0;
                for (unsigned short i = 0; i < sz; ++i) {
                    hnswlib::tableint v = data[i];
                    if (v >= cur_cnt) continue;
                    if (passes(v)) {
                        num_found++;
                        float d = dist_to_q(v);
                        if (d < curdist || !curr_pass) {
                            currObj = v;
                            curdist = d;
                            curr_pass = true;
                        }
                        if (num_found >= m_) break;
                    } else {
                        hnswlib::linklistsizeint* ll2 = hnsw_->get_linklist(v, level);
                        unsigned short sz2 = hnsw_->getListCount(ll2);
                        auto* data2 = reinterpret_cast<hnswlib::tableint*>(ll2 + 1);
                        for (unsigned short j = 0; j < sz2; ++j) {
                            hnswlib::tableint v2 = data2[j];
                            if (v2 >= cur_cnt) continue;
                            if (passes(v2)) {
                                num_found++;
                                float d2 = dist_to_q(v2);
                                if (d2 < curdist || !curr_pass) {
                                    currObj = v2;
                                    curdist = d2;
                                    curr_pass = true;
                                }
                                if (num_found >= m_) break;
                            }
                        }
                        if (num_found >= m_) break;
                    }
                }
                if (currObj == prev_nearest) break;
            }
        }

        // Stage 2: Base-layer (level 0) 2-hop predicate-subgraph beam search (matching ACORN.cpp search_from_candidates_2, gamma=1)
        thread_local std::vector<uint32_t> vis_dist;
        thread_local std::vector<uint32_t> vis_hop1;
        thread_local uint32_t tok = 1;
        if (vis_dist.size() < cur_cnt) {
            vis_dist.assign(cur_cnt, 0);
            vis_hop1.assign(cur_cnt, 0);
        }
        if (++tok == 0) {
            std::fill(vis_dist.begin(), vis_dist.end(), 0);
            std::fill(vis_hop1.begin(), vis_hop1.end(), 0);
            tok = 1;
        }

        const std::size_t ef = std::max<std::size_t>(k, ef_search);
        using Pair = std::pair<float, hnswlib::tableint>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> top_valid;

        vis_dist[currObj] = tok;
        cand_min.emplace(curdist, currObj);
        if (curr_pass) {
            top_valid.emplace(curdist, currObj);
        }

        const uint32_t m0_cap = m_ * 2;
        while (!cand_min.empty()) {
            auto [cd, cid] = cand_min.top();
            cand_min.pop();
            if (top_valid.size() >= ef && cd > top_valid.top().first) break;

            hnswlib::linklistsizeint* ll0 = hnsw_->get_linklist0(cid);
            unsigned short sz = hnsw_->getListCount(ll0);
            auto* data = reinterpret_cast<hnswlib::tableint*>(ll0 + 1);
            uint32_t found_here = 0;

            for (unsigned short i = 0; i < sz; ++i) {
                hnswlib::tableint v = data[i];
                if (v >= cur_cnt) continue;
                if (vis_dist[v] == tok) continue;

                if (passes(v)) {
                    vis_dist[v] = tok;
                    found_here++;
                    float d = dist_to_q(v);
                    if (top_valid.size() < ef || d < top_valid.top().first) {
                        cand_min.emplace(d, v);
                        top_valid.emplace(d, v);
                        if (top_valid.size() > ef) top_valid.pop();
                    }
                } else if (vis_hop1[v] != tok) {
                    vis_hop1[v] = tok;
                    hnswlib::linklistsizeint* ll2 = hnsw_->get_linklist0(v);
                    unsigned short sz2 = hnsw_->getListCount(ll2);
                    auto* data2 = reinterpret_cast<hnswlib::tableint*>(ll2 + 1);
                    for (unsigned short j = 0; j < sz2; ++j) {
                        hnswlib::tableint v2 = data2[j];
                        if (v2 >= cur_cnt || vis_dist[v2] == tok) continue;
                        if (passes(v2)) {
                            vis_dist[v2] = tok;
                            found_here++;
                            float d2 = dist_to_q(v2);
                            if (top_valid.size() < ef || d2 < top_valid.top().first) {
                                cand_min.emplace(d2, v2);
                                top_valid.emplace(d2, v2);
                                if (top_valid.size() > ef) top_valid.pop();
                            }
                            if (found_here >= m0_cap) break;
                        }
                    }
                }
                if (found_here >= m0_cap) break;
            }
        }

        while (top_valid.size() > k) top_valid.pop();
        out.ids.resize(top_valid.size());
        out.distances.resize(top_valid.size());
        for (std::size_t idx = top_valid.size(); idx > 0; --idx) {
            auto [d, iid] = top_valid.top();
            top_valid.pop();
            std::size_t lbl = static_cast<std::size_t>(hnsw_->getExternalLabel(iid));
            out.distances[idx - 1] = d;
            out.ids[idx - 1] = ext_ids_[lbl];
        }
        auto t1 = std::chrono::steady_clock::now();
        out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }

    std::size_t memory_bytes() const noexcept {
        std::size_t b = ext_ids_.capacity() * sizeof(uint64_t)
                      + (tv_s_.size + tv_e_.size + tk_s_.size + tk_e_.size) * sizeof(uint32_t)
                      + ext_to_int_.bucket_count() * sizeof(void*)
                      + ext_to_int_.size() * (sizeof(void*) + sizeof(uint64_t) + sizeof(uint32_t) + 16u);
        if (hnsw_) b += hnsw_->indexFileSize();
        return b;
    }

private:
    std::size_t dim_;
    uint32_t m_, ef_c_;
    std::size_t n_ = 0;
    mutable hnswlib::L2Space space_;
    mutable std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
    std::vector<uint64_t> ext_ids_;
    std::unordered_map<uint64_t, uint32_t> ext_to_int_;
    AlignedBuffer<uint32_t, 64> tv_s_, tv_e_, tk_s_, tk_e_;
};

// Baseline 1: Standard Global HNSW with Predicate / Bitmap Filtering (Industrial Vector DB Baseline)
class HNSWPostIndex {
public:
    HNSWPostIndex(std::size_t dim, uint32_t m = 16, uint32_t ef_construction = 96)
        : dim_(dim), m_(m), ef_c_(ef_construction) {}

    void build_batch(
        const uint64_t* ext_ids,
        const float* vectors,
        const uint32_t* tv_s,
        const uint32_t* tv_e,
        const uint32_t* tk_s,
        const uint32_t* tk_e,
        std::size_t n,
        int num_threads = 0) {
        if (n == 0) return;
        std::size_t base = n_;
        n_ += n;
        ext_ids_.resize(n_);
        tv_s_.resize(n_);
        tv_e_.resize(n_);
        tk_s_.resize(n_);
        tk_e_.resize(n_);
        vecs_.resize(n_ * dim_);
        adj_.resize(n_);
        std::vector<SpinLock> new_locks(n_);
        locks_.swap(new_locks);

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t id = base + i;
            ext_ids_[id] = ext_ids[i];
            ext_to_int_[ext_ids[i]] = static_cast<uint32_t>(id);
            tv_s_[id] = tv_s[i];
            tv_e_[id] = tv_e[i];
            tk_s_[id] = tk_s[i];
            tk_e_[id] = tk_e[i];
            std::memcpy(vecs_.data + id * dim_, vectors + i * dim_, dim_ * sizeof(float));
            adj_[id].reserve(m_ * 2);
        }

        #ifdef _OPENMP
        int prev_t = omp_get_max_threads();
        if (num_threads > 0) omp_set_num_threads(num_threads);
        #endif

        std::size_t start_i = (base == 0) ? 1 : 0;
        #pragma omp parallel if(n > 64)
        {
            std::vector<uint32_t> vis(n_, 0);
            uint32_t tok = 1;
            #pragma omp for schedule(dynamic, 32)
            for (std::size_t i = start_i; i < n; ++i) {
                uint32_t u = static_cast<uint32_t>(base + i);
                if (++tok == 0) { std::fill(vis.begin(), vis.end(), 0); tok = 1; }
                link_standard(u, vis, tok);
            }
        }

        #ifdef _OPENMP
        if (num_threads > 0) omp_set_num_threads(prev_t);
        #endif
    }

    void update_tx_end(uint64_t ext_id, uint32_t new_tk_e) {
        auto it = ext_to_int_.find(ext_id);
        if (it != ext_to_int_.end()) tk_e_[it->second] = new_tk_e;
    }

    void insert_single(
        uint64_t ext_id, const float* vec,
        uint32_t tv_s, uint32_t tv_e, uint32_t tk_s, uint32_t tk_e) {
        build_batch(&ext_id, vec, &tv_s, &tv_e, &tk_s, &tk_e, 1, 1);
    }

    QueryResult search(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t ef_search = 64) const {
        auto t0 = std::chrono::steady_clock::now();
        QueryResult out;
        if (n_ == 0 || k == 0) return out;

        thread_local std::vector<uint32_t> vis;
        thread_local uint32_t tok = 1;
        if (vis.size() < n_) vis.assign(n_, 0);
        if (++tok == 0) { std::fill(vis.begin(), vis.end(), 0); tok = 1; }

        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> beam_top;
        std::priority_queue<Pair> valid_top;

        // Standard HNSW starts at global entry points (unaware of temporal prefix!)
        const uint32_t step = static_cast<uint32_t>(std::max<std::size_t>(1, n_ / 8));
        for (uint32_t ep = 0; ep < n_ && cand_min.size() < 8; ep += step) {
            vis[ep] = tok;
            float d = simd::l2_sqr_fp32(q_vec, vecs_.data + static_cast<std::size_t>(ep) * dim_, dim_);
            out.stats.distance_computations++;
            cand_min.emplace(d, ep);
            beam_top.emplace(d, ep);
            out.stats.predicate_checks++;
            if (bitemp_visible(tv_s_[ep], tv_e_[ep], tk_s_[ep], tk_e_[ep], tv_q, tk_q)) {
                valid_top.emplace(d, ep);
                if (valid_top.size() > k) valid_top.pop();
            }
        }

        while (!cand_min.empty()) {
            auto [cd, curr] = cand_min.top();
            cand_min.pop();
            if (beam_top.size() >= ef_search && cd > beam_top.top().first) break;

            for (uint32_t nb : adj_[curr]) {
                if (vis[nb] == tok) continue;
                vis[nb] = tok;
                float d = simd::l2_sqr_fp32(q_vec, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_);
                out.stats.distance_computations++;

                out.stats.predicate_checks++;
                if (bitemp_visible(tv_s_[nb], tv_e_[nb], tk_s_[nb], tk_e_[nb], tv_q, tk_q)) {
                    if (valid_top.size() < k) valid_top.emplace(d, nb);
                    else if (d < valid_top.top().first) { valid_top.pop(); valid_top.emplace(d, nb); }
                }
                if (beam_top.size() < ef_search || d < beam_top.top().first) {
                    cand_min.emplace(d, nb);
                    beam_top.emplace(d, nb);
                    if (beam_top.size() > ef_search) beam_top.pop();
                }
            }
        }

        out.ids.resize(valid_top.size());
        out.distances.resize(valid_top.size());
        for (std::size_t idx = valid_top.size(); idx > 0; --idx) {
            out.ids[idx - 1] = ext_ids_[valid_top.top().second];
            out.distances[idx - 1] = valid_top.top().first;
            valid_top.pop();
        }
        auto t1 = std::chrono::steady_clock::now();
        out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }


    std::size_t memory_bytes() const noexcept {
        std::size_t b = ext_ids_.capacity() * sizeof(uint64_t)
                      + (tv_s_.size + tv_e_.size + tk_s_.size + tk_e_.size) * sizeof(uint32_t)
                      + vecs_.size * sizeof(float)
                      + ext_to_int_.bucket_count() * sizeof(void*)
                      + ext_to_int_.size() * (sizeof(void*) + sizeof(uint64_t) + sizeof(uint32_t) + 16u)
                      + adj_.capacity() * sizeof(std::vector<uint32_t>);
        for (const auto& a : adj_) b += a.capacity() * sizeof(uint32_t);
        return b;
    }

private:
    std::size_t dim_;
    uint32_t m_, ef_c_;
    std::size_t n_ = 0;
    std::vector<uint64_t> ext_ids_;
    std::unordered_map<uint64_t, uint32_t> ext_to_int_;
    AlignedBuffer<uint32_t, 64> tv_s_, tv_e_, tk_s_, tk_e_;
    AlignedBuffer<float, 64> vecs_;
    std::vector<std::vector<uint32_t>> adj_;
    std::vector<SpinLock> locks_;

    void link_standard(uint32_t u, std::vector<uint32_t>& vis, uint32_t tok) {
        const float* uv = vecs_.data + static_cast<std::size_t>(u) * dim_;
        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> top_ef;
        vis[u] = tok;

        // Seed from earlier inserted nodes
        uint32_t limit = u;
        uint32_t stride = std::max<uint32_t>(1, limit / 6);
        for (uint32_t s = 0; s < limit && cand_min.size() < 6; s += stride) {
            vis[s] = tok;
            float d = simd::l2_sqr_fp32(uv, vecs_.data + static_cast<std::size_t>(s) * dim_, dim_);
            cand_min.emplace(d, s);
            top_ef.emplace(d, s);
        }

        uint32_t exp = 0;
        while (!cand_min.empty() && exp < ef_c_ * 3) {
            auto [cd, curr] = cand_min.top();
            cand_min.pop();
            if (top_ef.size() >= ef_c_ && cd > top_ef.top().first) break;
            exp++;

            uint32_t nbs[48];
            uint32_t cnt = 0;
            locks_[curr].lock();
            for (uint32_t w : adj_[curr]) { if (cnt < 48) nbs[cnt++] = w; }
            locks_[curr].unlock();

            for (uint32_t j = 0; j < cnt; ++j) {
                uint32_t nb = nbs[j];
                if (nb >= u || vis[nb] == tok) continue;
                vis[nb] = tok;
                float d = simd::l2_sqr_fp32(uv, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_);
                if (top_ef.size() < ef_c_ || d < top_ef.top().first) {
                    cand_min.emplace(d, nb);
                    top_ef.emplace(d, nb);
                    if (top_ef.size() > ef_c_) top_ef.pop();
                }
            }
        }

        std::vector<Pair> cands;
        cands.reserve(top_ef.size());
        while (!top_ef.empty()) { cands.push_back(top_ef.top()); top_ef.pop(); }
        std::sort(cands.begin(), cands.end());

        // RNG-diversified neighbor selection (Malkov & Yashunin HNSW Algorithm 4 heuristic)
        std::vector<uint32_t> selected;
        selected.reserve(m_);
        for (const auto& [cand_d, cand_id] : cands) {
            if (selected.size() >= m_) break;
            const float* cv = vecs_.data + static_cast<std::size_t>(cand_id) * dim_;
            bool keep = true;
            for (uint32_t sel_id : selected) {
                float d_cs = simd::l2_sqr_fp32(cv, vecs_.data + static_cast<std::size_t>(sel_id) * dim_, dim_);
                if (d_cs < cand_d) { keep = false; break; }
            }
            if (keep) selected.push_back(cand_id);
        }
        for (const auto& [cand_d, cand_id] : cands) {
            if (selected.size() >= m_) break;
            if (std::find(selected.begin(), selected.end(), cand_id) == selected.end()) {
                selected.push_back(cand_id);
            }
        }

        locks_[u].lock();
        for (uint32_t w : selected) adj_[u].push_back(w);
        locks_[u].unlock();

        for (uint32_t w : selected) {
            uint32_t cur_nbs[64];
            std::size_t cur_sz = 0;
            bool fast_pushed = false;
            locks_[w].lock();
            if (adj_[w].size() < m_ * 2) {
                adj_[w].push_back(u);
                fast_pushed = true;
            } else {
                cur_sz = std::min<std::size_t>(64, adj_[w].size());
                for (std::size_t k_i = 0; k_i < cur_sz; ++k_i) cur_nbs[k_i] = adj_[w][k_i];
            }
            locks_[w].unlock();
            if (fast_pushed) continue;

            // RNG-diversified degree pruning computed lock-free outside spinlock
            const float* wv = vecs_.data + static_cast<std::size_t>(w) * dim_;
            std::vector<Pair> pool;
            pool.reserve(cur_sz + 1);
            for (std::size_t k_i = 0; k_i < cur_sz; ++k_i) {
                uint32_t nb = cur_nbs[k_i];
                pool.emplace_back(simd::l2_sqr_fp32(wv, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_), nb);
            }
            pool.emplace_back(simd::l2_sqr_fp32(wv, uv, dim_), u);
            std::sort(pool.begin(), pool.end());
            std::vector<uint32_t> pruned;
            pruned.reserve(m_ * 2);
            for (const auto& [pd, pid] : pool) {
                if (pruned.size() >= m_ * 2) break;
                const float* pv = vecs_.data + static_cast<std::size_t>(pid) * dim_;
                bool keep = true;
                for (uint32_t sid : pruned) {
                    if (simd::l2_sqr_fp32(pv, vecs_.data + static_cast<std::size_t>(sid) * dim_, dim_) < pd) {
                        keep = false;
                        break;
                    }
                }
                if (keep) pruned.push_back(pid);
            }
            for (const auto& [pd, pid] : pool) {
                if (pruned.size() >= m_ * 2) break;
                if (std::find(pruned.begin(), pruned.end(), pid) == pruned.end()) pruned.push_back(pid);
            }
            locks_[w].lock();
            adj_[w].swap(pruned);
            locks_[w].unlock();
        }
    }
};

// Baseline 2 & 3: 1D Segment Graph Family
// axis_mode == 0 => SeRF-Adapt (sorted static 1D segment graph on Valid Start tv_s, SIGMOD 2024)
// axis_mode == 1 => DSG-1D     (streaming 1D dynamic segment graph on Tx Start tk_s, PVLDB 2025)
class OneDimSegmentGraphIndex {
public:
    OneDimSegmentGraphIndex(std::size_t dim, int axis_mode = 0, uint32_t m = 16, uint32_t ef_c = 96)
        : dim_(dim), axis_mode_(axis_mode), m_(m), ef_c_(ef_c) {}

    void build_batch(
        const uint64_t* ext_ids,
        const float* vectors,
        const uint32_t* tv_s,
        const uint32_t* tv_e,
        const uint32_t* tk_s,
        const uint32_t* tk_e,
        std::size_t n,
        int num_threads = 0) {
        if (n == 0) return;
        // Sort by chosen 1D primary axis (tv_s for SeRF-Adapt, tk_s for DSG-1D)
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        const uint32_t* primary = (axis_mode_ == 0) ? tv_s : tk_s;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return primary[a] < primary[b];
        });

        n_ = n;
        ext_ids_.resize(n_);
        key_sorted_.resize(n_);
        tv_s_.resize(n_);
        tv_e_.resize(n_);
        tk_s_.resize(n_);
        tk_e_.resize(n_);
        vecs_.resize(n_ * dim_);
        adj_left_.assign(n_, {});
        adj_right_.assign(n_, {});
        std::vector<SpinLock> new_locks(n_);
        locks_.swap(new_locks);

        for (std::size_t rank = 0; rank < n_; ++rank) {
            std::size_t src = order[rank];
            ext_ids_[rank] = ext_ids[src];
            ext_to_int_[ext_ids[src]] = static_cast<uint32_t>(rank);
            key_sorted_[rank] = primary[src];
            tv_s_[rank] = tv_s[src];
            tv_e_[rank] = tv_e[src];
            tk_s_[rank] = tk_s[src];
            tk_e_[rank] = tk_e[src];
            std::memcpy(vecs_.data + rank * dim_, vectors + src * dim_, dim_ * sizeof(float));
            adj_left_[rank].reserve(m_);
            adj_right_[rank].reserve(m_);
        }

        #ifdef _OPENMP
        int prev_t = omp_get_max_threads();
        if (num_threads > 0) omp_set_num_threads(num_threads);
        #endif

        // Build prefix-compressed 1D segment edges along the primary scalar axis
        #pragma omp parallel if(n_ > 64)
        {
            std::vector<uint32_t> vis(n_, 0);
            uint32_t tok = 1;
            #pragma omp for schedule(dynamic, 32)
            for (std::size_t r = 1; r < n_; ++r) {
                if (++tok == 0) { std::fill(vis.begin(), vis.end(), 0); tok = 1; }
                link_1d_prefix(static_cast<uint32_t>(r), vis, tok);
            }
        }

        #ifdef _OPENMP
        if (num_threads > 0) omp_set_num_threads(prev_t);
        #endif
    }

    void update_tx_end(uint64_t ext_id, uint32_t new_tk_e) {
        auto it = ext_to_int_.find(ext_id);
        if (it != ext_to_int_.end()) tk_e_[it->second] = new_tk_e;
    }

    // Search restricts 1D prefix [0, upper_rank(q_primary)] natively,
    // and post-filters the remaining 3 bi-temporal bounds!
    QueryResult search(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t ef_search = 64) const {
        auto t0 = std::chrono::steady_clock::now();
        QueryResult out;
        if (n_ == 0 || k == 0) return out;

        uint32_t q_key = (axis_mode_ == 0) ? tv_q : tk_q;
        auto ub = std::upper_bound(key_sorted_.begin(), key_sorted_.end(), q_key);
        std::size_t max_rank = static_cast<std::size_t>(ub - key_sorted_.begin());
        if (max_rank == 0) {
            auto t1 = std::chrono::steady_clock::now();
            out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            return out;
        }

        thread_local std::vector<uint32_t> vis;
        thread_local uint32_t tok = 1;
        if (vis.size() < n_) vis.assign(n_, 0);
        if (++tok == 0) { std::fill(vis.begin(), vis.end(), 0); tok = 1; }

        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> beam_top;
        std::priority_queue<Pair> valid_top;

        uint32_t step = static_cast<uint32_t>(std::max<std::size_t>(1, max_rank / 8));
        for (uint32_t ep = static_cast<uint32_t>(max_rank - 1);; ep -= step) {
            vis[ep] = tok;
            float d = simd::l2_sqr_fp32(q_vec, vecs_.data + static_cast<std::size_t>(ep) * dim_, dim_);
            out.stats.distance_computations++;
            cand_min.emplace(d, ep);
            beam_top.emplace(d, ep);
            out.stats.predicate_checks++;
            if (bitemp_visible(tv_s_[ep], tv_e_[ep], tk_s_[ep], tk_e_[ep], tv_q, tk_q)) {
                valid_top.emplace(d, ep);
                if (valid_top.size() > k) valid_top.pop();
            }
            if (ep < step || cand_min.size() >= 8) break;
        }

        while (!cand_min.empty()) {
            auto [cd, curr] = cand_min.top();
            cand_min.pop();
            if (beam_top.size() >= ef_search && cd > beam_top.top().first) break;

            for (uint32_t nb : adj_left_[curr]) {
                if (vis[nb] == tok) continue;
                vis[nb] = tok;
                float d = simd::l2_sqr_fp32(q_vec, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_);
                out.stats.distance_computations++;
                out.stats.predicate_checks++;
                if (bitemp_visible(tv_s_[nb], tv_e_[nb], tk_s_[nb], tk_e_[nb], tv_q, tk_q)) {
                    if (valid_top.size() < k) valid_top.emplace(d, nb);
                    else if (d < valid_top.top().first) { valid_top.pop(); valid_top.emplace(d, nb); }
                }
                if (beam_top.size() < ef_search || d < beam_top.top().first) {
                    cand_min.emplace(d, nb);
                    beam_top.emplace(d, nb);
                    if (beam_top.size() > ef_search) beam_top.pop();
                }
            }
            for (uint32_t nb : adj_right_[curr]) {
                if (nb >= max_rank || vis[nb] == tok) continue; // 1D segment bound check
                vis[nb] = tok;
                float d = simd::l2_sqr_fp32(q_vec, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_);
                out.stats.distance_computations++;
                out.stats.predicate_checks++;
                if (bitemp_visible(tv_s_[nb], tv_e_[nb], tk_s_[nb], tk_e_[nb], tv_q, tk_q)) {
                    if (valid_top.size() < k) valid_top.emplace(d, nb);
                    else if (d < valid_top.top().first) { valid_top.pop(); valid_top.emplace(d, nb); }
                }
                if (beam_top.size() < ef_search || d < beam_top.top().first) {
                    cand_min.emplace(d, nb);
                    beam_top.emplace(d, nb);
                    if (beam_top.size() > ef_search) beam_top.pop();
                }
            }
        }

        out.ids.resize(valid_top.size());
        out.distances.resize(valid_top.size());
        for (std::size_t idx = valid_top.size(); idx > 0; --idx) {
            out.ids[idx - 1] = ext_ids_[valid_top.top().second];
            out.distances[idx - 1] = valid_top.top().first;
            valid_top.pop();
        }
        auto t1 = std::chrono::steady_clock::now();
        out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }


private:
    std::size_t dim_;
    int axis_mode_;
    uint32_t m_, ef_c_;
    std::size_t n_ = 0;
    std::vector<uint64_t> ext_ids_;
    std::vector<uint32_t> key_sorted_;
    std::unordered_map<uint64_t, uint32_t> ext_to_int_;
    AlignedBuffer<uint32_t, 64> tv_s_, tv_e_, tk_s_, tk_e_;
    AlignedBuffer<float, 64> vecs_;
    std::vector<std::vector<uint32_t>> adj_left_;
    std::vector<std::vector<uint32_t>> adj_right_;
    std::vector<SpinLock> locks_;

    void link_1d_prefix(uint32_t r, std::vector<uint32_t>& vis, uint32_t tok) {
        const float* rv = vecs_.data + static_cast<std::size_t>(r) * dim_;
        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> top_ef;
        vis[r] = tok;

        uint32_t step = std::max<uint32_t>(1, r / 6);
        for (uint32_t s = r - 1;; s -= step) {
            vis[s] = tok;
            float d = simd::l2_sqr_fp32(rv, vecs_.data + static_cast<std::size_t>(s) * dim_, dim_);
            cand_min.emplace(d, s);
            top_ef.emplace(d, s);
            if (s < step || cand_min.size() >= 6) break;
        }

        uint32_t exp = 0;
        while (!cand_min.empty() && exp < ef_c_ * 2) {
            auto [cd, curr] = cand_min.top();
            cand_min.pop();
            if (top_ef.size() >= ef_c_ && cd > top_ef.top().first) break;
            exp++;

            uint32_t nbs[32];
            uint32_t cnt = 0;
            locks_[curr].lock();
            for (uint32_t w : adj_left_[curr]) { if (cnt < 32) nbs[cnt++] = w; }
            for (uint32_t w : adj_right_[curr]) { if (w < r && cnt < 32) nbs[cnt++] = w; }
            locks_[curr].unlock();

            for (uint32_t j = 0; j < cnt; ++j) {
                uint32_t nb = nbs[j];
                if (vis[nb] == tok) continue;
                vis[nb] = tok;
                float d = simd::l2_sqr_fp32(rv, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_);
                if (top_ef.size() < ef_c_ || d < top_ef.top().first) {
                    cand_min.emplace(d, nb);
                    top_ef.emplace(d, nb);
                    if (top_ef.size() > ef_c_) top_ef.pop();
                }
            }
        }

        std::vector<Pair> cands;
        while (!top_ef.empty()) { cands.push_back(top_ef.top()); top_ef.pop(); }
        std::sort(cands.begin(), cands.end());
        uint32_t take = std::min<uint32_t>(m_, static_cast<uint32_t>(cands.size()));

        locks_[r].lock();
        for (uint32_t i = 0; i < take; ++i) adj_left_[r].push_back(cands[i].second);
        locks_[r].unlock();

        for (uint32_t i = 0; i < take; ++i) {
            uint32_t w = cands[i].second;
            locks_[w].lock();
            if (adj_right_[w].size() < m_) adj_right_[w].push_back(r);
            locks_[w].unlock();
        }
    }

public:
    std::size_t memory_bytes() const noexcept {
        std::size_t b = ext_ids_.capacity() * sizeof(uint64_t)
            + key_sorted_.capacity() * sizeof(uint32_t)
            + (tv_s_.size + tv_e_.size + tk_s_.size + tk_e_.size) * sizeof(uint32_t)
            + vecs_.size * sizeof(float)
            + ext_to_int_.bucket_count() * sizeof(void*)
            + ext_to_int_.size() * (sizeof(void*) + sizeof(uint64_t) + sizeof(uint32_t) + 16u)
            + (adj_left_.capacity() + adj_right_.capacity()) * sizeof(std::vector<uint32_t>);
        for (const auto& a : adj_left_) b += a.capacity() * sizeof(uint32_t);
        for (const auto& a : adj_right_) b += a.capacity() * sizeof(uint32_t);
        return b;
    }
};

// Baseline 4: ACORN-BiTemp (Predicate-Agnostic Expanded HNSW + 2-Hop Predicate-Subgraph Traversal, Patel et al. SIGMOD 2024)
class ACORNIndex {
public:
    ACORNIndex(std::size_t dim, uint32_t m = 16, uint32_t gamma = 2, uint32_t ef_c = 96)
        : dim_(dim), m_(m), m_beta_(m * std::max<uint32_t>(1, gamma)), ef_c_(ef_c) {}

    void build_batch(
        const uint64_t* ext_ids,
        const float* vectors,
        const uint32_t* tv_s,
        const uint32_t* tv_e,
        const uint32_t* tk_s,
        const uint32_t* tk_e,
        std::size_t n,
        int num_threads = 0) {
        if (n == 0) return;
        std::size_t base = n_;
        n_ += n;
        ext_ids_.resize(n_);
        tv_s_.resize(n_);
        tv_e_.resize(n_);
        tk_s_.resize(n_);
        tk_e_.resize(n_);
        vecs_.resize(n_ * dim_);
        adj_.resize(n_);
        if (locks_.size() < n_) {
            std::vector<SpinLock> nl(std::max<std::size_t>(n_, locks_.empty() ? n_ : locks_.size() * 2));
            locks_.swap(nl);
        }

        for (std::size_t i = 0; i < n; ++i) {
            uint32_t u = static_cast<uint32_t>(base + i);
            ext_ids_[u] = ext_ids[i];
            ext_to_int_[ext_ids[i]] = u;
            tv_s_[u] = tv_s[i];
            tv_e_[u] = tv_e[i];
            tk_s_[u] = tk_s[i];
            tk_e_[u] = tk_e[i];
            std::memcpy(vecs_.data + static_cast<std::size_t>(u) * dim_, vectors + i * dim_, dim_ * sizeof(float));
            adj_[u].reserve(m_beta_ * 2);
        }

        #ifdef _OPENMP
        int prev_t = omp_get_max_threads();
        if (num_threads > 0) omp_set_num_threads(num_threads);
        #endif

        std::size_t boot = std::min<std::size_t>(n, 16);
        {
            std::vector<uint32_t> vis(n_, 0);
            uint32_t tok = 1;
            for (std::size_t i = 0; i < boot; ++i) {
                uint32_t u = static_cast<uint32_t>(base + i);
                if (u == 0) continue;
                if (++tok == 0) { std::fill(vis.begin(), vis.end(), 0); tok = 1; }
                link_acorn(u, vis, tok);
            }
        }

        #pragma omp parallel if(n > 64)
        {
            std::vector<uint32_t> vis(n_, 0);
            uint32_t tok = 1;
            #pragma omp for schedule(dynamic, 32)
            for (std::size_t i = boot; i < n; ++i) {
                uint32_t u = static_cast<uint32_t>(base + i);
                if (++tok == 0) { std::fill(vis.begin(), vis.end(), 0); tok = 1; }
                link_acorn(u, vis, tok);
            }
        }

        #ifdef _OPENMP
        if (num_threads > 0) omp_set_num_threads(prev_t);
        #endif
    }

    void update_tx_end(uint64_t ext_id, uint32_t new_tk_e) {
        auto it = ext_to_int_.find(ext_id);
        if (it != ext_to_int_.end()) tk_e_[it->second] = new_tk_e;
    }

    QueryResult search(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t ef_search = 64) const {
        auto t0 = std::chrono::steady_clock::now();
        QueryResult out;
        if (n_ == 0 || k == 0) return out;

        thread_local std::vector<uint32_t> vis_dist;
        thread_local std::vector<uint32_t> vis_pred;
        thread_local uint32_t tok = 1;
        if (vis_dist.size() < n_) {
            vis_dist.assign(n_, 0);
            vis_pred.assign(n_, 0);
        }
        if (++tok == 0) {
            std::fill(vis_dist.begin(), vis_dist.end(), 0);
            std::fill(vis_pred.begin(), vis_pred.end(), 0);
            tok = 1;
        }

        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> valid_top;

        // Predicate-aware entry point scan (probes up to 256 stride positions to locate initial predicate-passing seeds)
        const std::size_t probe_cap = std::min<std::size_t>(n_, 256);
        const std::size_t step = std::max<std::size_t>(1, n_ / probe_cap);
        for (std::size_t idx = 0; idx < n_ && cand_min.size() < 8; idx += step) {
            uint32_t ep = static_cast<uint32_t>(idx);
            vis_pred[ep] = tok;
            out.stats.predicate_checks++;
            if (bitemp_visible(tv_s_[ep], tv_e_[ep], tk_s_[ep], tk_e_[ep], tv_q, tk_q)) {
                vis_dist[ep] = tok;
                float d = simd::l2_sqr_fp32(q_vec, vecs_.data + static_cast<std::size_t>(ep) * dim_, dim_);
                out.stats.distance_computations++;
                cand_min.emplace(d, ep);
                valid_top.emplace(d, ep);
                if (valid_top.size() > ef_search) valid_top.pop();
            }
        }
        // Fallback to standard global entry point if ultra-sparse predicate had 0 hits in initial probe
        if (cand_min.empty()) {
            uint32_t ep = 0;
            vis_dist[ep] = tok;
            float d = simd::l2_sqr_fp32(q_vec, vecs_.data, dim_);
            out.stats.distance_computations++;
            cand_min.emplace(d, ep);
        }

        std::vector<uint32_t> pass_nbs;
        std::vector<uint32_t> fail_nbs;
        pass_nbs.reserve(m_beta_ * 2);
        fail_nbs.reserve(m_beta_ * 2);

        uint32_t expansions = 0;
        const uint32_t max_exp = ef_search * 8;

        while (!cand_min.empty() && expansions < max_exp) {
            auto [cd, curr] = cand_min.top();
            cand_min.pop();
            if (valid_top.size() >= ef_search && cd > valid_top.top().first) break;
            expansions++;

            pass_nbs.clear();
            fail_nbs.clear();

            // 1-hop predicate filter BEFORE vector distance evaluation
            for (uint32_t w : adj_[curr]) {
                if (vis_dist[w] == tok) continue;
                out.stats.predicate_checks++;
                if (bitemp_visible(tv_s_[w], tv_e_[w], tk_s_[w], tk_e_[w], tv_q, tk_q)) {
                    vis_dist[w] = tok;
                    pass_nbs.push_back(w);
                } else {
                    fail_nbs.push_back(w);
                }
            }

            // ACORN 2-hop predicate subgraph expansion when 1-hop visible degree < M
            if (pass_nbs.size() < m_) {
                for (uint32_t w : fail_nbs) {
                    if (pass_nbs.size() >= m_ * 2) break;
                    for (uint32_t u : adj_[w]) {
                        if (vis_dist[u] == tok || vis_pred[u] == tok) continue;
                        vis_pred[u] = tok;
                        out.stats.predicate_checks++;
                        if (bitemp_visible(tv_s_[u], tv_e_[u], tk_s_[u], tk_e_[u], tv_q, tk_q)) {
                            vis_dist[u] = tok;
                            pass_nbs.push_back(u);
                            if (pass_nbs.size() >= m_ * 2) break;
                        }
                    }
                }
            }

            for (uint32_t nb : pass_nbs) {
                float d = simd::l2_sqr_fp32(q_vec, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_);
                out.stats.distance_computations++;
                if (valid_top.size() < ef_search || d < valid_top.top().first) {
                    cand_min.emplace(d, nb);
                    valid_top.emplace(d, nb);
                    if (valid_top.size() > ef_search) valid_top.pop();
                }
            }

            // If 2-hop neighborhood still yielded 0 visible neighbors (percolation collapse at low sigma),
            // allow stepping across at most 2 unvisited 1-hop bridges so traversal does not halt on step 1
            if (pass_nbs.empty() && valid_top.size() < k) {
                uint32_t bridge_steps = 0;
                for (uint32_t w : fail_nbs) {
                    if (bridge_steps >= 2) break;
                    if (vis_dist[w] == tok) continue;
                    vis_dist[w] = tok;
                    float d = simd::l2_sqr_fp32(q_vec, vecs_.data + static_cast<std::size_t>(w) * dim_, dim_);
                    out.stats.distance_computations++;
                    cand_min.emplace(d, w);
                    bridge_steps++;
                }
            }
        }

        std::vector<Pair> hits;
        hits.reserve(valid_top.size());
        while (!valid_top.empty()) {
            hits.push_back(valid_top.top());
            valid_top.pop();
        }
        std::sort(hits.begin(), hits.end());
        std::size_t take = std::min<std::size_t>(k, hits.size());
        out.ids.resize(take);
        out.distances.resize(take);
        for (std::size_t i = 0; i < take; ++i) {
            out.ids[i] = ext_ids_[hits[i].second];
            out.distances[i] = hits[i].first;
        }
        auto t1 = std::chrono::steady_clock::now();
        out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }


    std::size_t memory_bytes() const noexcept {
        std::size_t b = ext_ids_.capacity() * sizeof(uint64_t)
                      + (tv_s_.size + tv_e_.size + tk_s_.size + tk_e_.size) * sizeof(uint32_t)
                      + vecs_.size * sizeof(float)
                      + ext_to_int_.bucket_count() * sizeof(void*)
                      + ext_to_int_.size() * (sizeof(void*) + sizeof(uint64_t) + sizeof(uint32_t) + 16u)
                      + adj_.capacity() * sizeof(std::vector<uint32_t>);
        for (const auto& a : adj_) b += a.capacity() * sizeof(uint32_t);
        return b;
    }

private:
    std::size_t dim_;
    uint32_t m_, m_beta_, ef_c_;
    std::size_t n_ = 0;
    std::vector<uint64_t> ext_ids_;
    std::unordered_map<uint64_t, uint32_t> ext_to_int_;
    AlignedBuffer<uint32_t, 64> tv_s_, tv_e_, tk_s_, tk_e_;
    AlignedBuffer<float, 64> vecs_;
    std::vector<std::vector<uint32_t>> adj_;
    std::vector<SpinLock> locks_;

    void link_acorn(uint32_t u, std::vector<uint32_t>& vis, uint32_t tok) {
        const float* uv = vecs_.data + static_cast<std::size_t>(u) * dim_;
        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> top_ef;
        vis[u] = tok;

        uint32_t stride = std::max<uint32_t>(1, u / 6);
        for (uint32_t s = 0; s < u && cand_min.size() < 6; s += stride) {
            vis[s] = tok;
            float d = simd::l2_sqr_fp32(uv, vecs_.data + static_cast<std::size_t>(s) * dim_, dim_);
            cand_min.emplace(d, s);
            top_ef.emplace(d, s);
        }

        uint32_t exp = 0;
        while (!cand_min.empty() && exp < ef_c_ * 3) {
            auto [cd, curr] = cand_min.top();
            cand_min.pop();
            if (top_ef.size() >= ef_c_ && cd > top_ef.top().first) break;
            exp++;

            uint32_t nbs[64];
            uint32_t cnt = 0;
            locks_[curr].lock();
            for (uint32_t w : adj_[curr]) { if (cnt < 64) nbs[cnt++] = w; }
            locks_[curr].unlock();

            for (uint32_t j = 0; j < cnt; ++j) {
                uint32_t nb = nbs[j];
                if (nb >= u || vis[nb] == tok) continue;
                vis[nb] = tok;
                float d = simd::l2_sqr_fp32(uv, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_);
                if (top_ef.size() < ef_c_ || d < top_ef.top().first) {
                    cand_min.emplace(d, nb);
                    top_ef.emplace(d, nb);
                    if (top_ef.size() > ef_c_) top_ef.pop();
                }
            }
        }

        // ACORN keeps denser top-M_beta nearest candidates (light pruning) to preserve predicate subgraphs
        std::vector<Pair> cands;
        cands.reserve(top_ef.size());
        while (!top_ef.empty()) { cands.push_back(top_ef.top()); top_ef.pop(); }
        std::sort(cands.begin(), cands.end());
        uint32_t take = std::min<uint32_t>(m_beta_, static_cast<uint32_t>(cands.size()));

        locks_[u].lock();
        for (uint32_t i = 0; i < take; ++i) adj_[u].push_back(cands[i].second);
        locks_[u].unlock();

        for (uint32_t i = 0; i < take; ++i) {
            uint32_t w = cands[i].second;
            uint32_t cur_nbs[128];
            std::size_t cur_sz = 0;
            bool fast_pushed = false;
            locks_[w].lock();
            if (adj_[w].size() < m_beta_ * 2) {
                adj_[w].push_back(u);
                fast_pushed = true;
            } else {
                cur_sz = std::min<std::size_t>(128, adj_[w].size());
                for (std::size_t k_i = 0; k_i < cur_sz; ++k_i) cur_nbs[k_i] = adj_[w][k_i];
            }
            locks_[w].unlock();
            if (fast_pushed) continue;

            // Evaluate farthest neighbor lock-free outside spinlock
            const float* wv = vecs_.data + static_cast<std::size_t>(w) * dim_;
            float du = cands[i].first;
            float worst_d = -1.0f;
            std::size_t worst_idx = 0;
            for (std::size_t k_i = 0; k_i < cur_sz; ++k_i) {
                uint32_t nb = cur_nbs[k_i];
                float dn = simd::l2_sqr_fp32(wv, vecs_.data + static_cast<std::size_t>(nb) * dim_, dim_);
                if (dn > worst_d) { worst_d = dn; worst_idx = k_i; }
            }
            if (du < worst_d) {
                locks_[w].lock();
                if (worst_idx < adj_[w].size()) adj_[w][worst_idx] = u;
                locks_[w].unlock();
            }
        }
    }
};

// Baseline 5 (No-Graph Ablation): Grid-SQ8-Scan
// Streaming 2D Causal Birth-Grid + Mortality Histogram + Contiguous AVX2 SQ8 Scan + Exact FP32 Rerank,
// WITHOUT any Dual-Budget HNSW Graph construction or graph adjacency memory!
class GridSQ8ScanIndex {
public:
    explicit GridSQ8ScanIndex(std::size_t dim = 768, uint32_t grid_bins = 16, uint64_t initial_seg_cap = 131072)
        : dim_(dim), padded_dim_(((dim + 31) / 32) * 32), grid_bins_(grid_bins), initial_cap_(initial_seg_cap) {}

    void build_batch(
        const uint64_t* ext_ids,
        const float* vectors,
        const uint32_t* tv_s,
        const uint32_t* tv_e,
        const uint32_t* tk_s,
        const uint32_t* tk_e,
        std::size_t n,
        int /*num_threads*/ = 0) {
        if (n == 0) return;
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        if (!std::is_sorted(tk_s, tk_s + n)) {
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return tk_s[a] < tk_s[b];
            });
        }

        // Delegate cell partitioning to lightweight graph-free BiTHNSWConfig (m_pre=0 build bypass via scan_only)
        std::size_t cur = 0;
        while (cur < n) {
            if (segs_.empty() || segs_.back().num_nodes >= segs_.back().cap || segs_.back().active_gk >= grid_bins_) {
                uint64_t next_c = segs_.empty() ? initial_cap_ : std::min<uint64_t>(16000000ULL, segs_.back().cap * 2ULL);
                segs_.emplace_back(grid_bins_, next_c);
                calibrate_seg(segs_.back(), tv_s, tk_s, order.data() + cur, n - cur);
            }
            ScanSeg& seg = segs_.back();
            std::size_t take = std::min<std::size_t>(n - cur, static_cast<std::size_t>(seg.cap - seg.num_nodes));
            append_to_seg(seg, ext_ids, vectors, tv_s, tv_e, tk_s, tk_e, order.data() + cur, take);
            cur += take;
        }
    }

    QueryResult search(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t rerank_pool = 64) const {
        auto t0 = std::chrono::steady_clock::now();
        QueryResult out;
        if (total_nodes_ == 0 || k == 0) return out;

        simd::SQ8Query sq_q = simd::prepare_sq8_query(q_vec, dim_);
        const uint32_t cap_r = std::max<uint32_t>(k * 5, rerank_pool);
        using Cand = std::pair<float, uint64_t>;
        std::priority_queue<Cand> sq_top;

        for (std::size_t s_i = 0; s_i < segs_.size(); ++s_i) {
            const ScanSeg& seg = segs_[s_i];
            if (seg.num_nodes == 0) continue;
            if (seg.min_tv_s > tv_q || seg.min_tk_s > tk_q || seg.max_tv_e <= tv_q || seg.max_tk_e <= tk_q) continue;

            uint8_t q_gv = static_cast<uint8_t>(std::upper_bound(seg.v_bounds.begin(), seg.v_bounds.end(), tv_q) - seg.v_bounds.begin());
            uint8_t max_col = std::min<uint8_t>(seg.active_gk, static_cast<uint8_t>(grid_bins_ - 1));
            uint8_t q_gk = 0;
            for (uint8_t gk = 0; gk <= max_col; ++gk) {
                if (seg.gk_counts[gk] > 0 && seg.k_min[gk] <= tk_q) q_gk = gk;
            }
            uint64_t seg_pref = static_cast<uint64_t>(s_i) << 32;

            for (uint8_t gk = 0; gk <= q_gk; ++gk) {
                if (seg.k_min[gk] > tk_q) continue;
                for (uint8_t gv = 0; gv <= q_gv; ++gv) {
                    const ScanCell& cell = seg.cells[gk * grid_bins_ + gv];
                    std::size_t sz = cell.local_ids.size();
                    if (sz == 0) continue;
                    if (cell.min_tv_s > tv_q || cell.min_tk_s > tk_q || cell.max_tv_e <= tv_q || cell.max_tk_e <= tk_q) continue;

                    out.stats.visited_cells++;
                    out.stats.scan_cells++;
                    bool fb = (cell.max_tv_s <= tv_q) && (cell.max_tk_s <= tk_q);

                    for (std::size_t base = 0; base < sz; base += 8) {
                        uint8_t mask = fb
                            ? simd::filter8_death_only(cell.tv_e.data + base, cell.tk_e.data + base, tv_q, tk_q)
                            : simd::filter8_full_bitemp(
                                cell.tv_s.data + base, cell.tv_e.data + base,
                                cell.tk_s.data + base, cell.tk_e.data + base, tv_q, tk_q);
                        out.stats.predicate_checks += 8;
                        while (mask != 0) {
                            unsigned bit = static_cast<unsigned>(__builtin_ctz(mask));
                            mask &= static_cast<uint8_t>(mask - 1);
                            std::size_t slot = base + bit;
                            if (slot >= sz) break;
                            float ad = simd::sq8_approx_l2_sqr(
                                sq_q, cell.sq8_codes.data + slot * padded_dim_, cell.sq8_meta[slot]);
                            out.stats.distance_computations++;
                            uint64_t gh = seg_pref | static_cast<uint64_t>(cell.local_ids[slot]);
                            if (sq_top.size() < cap_r) sq_top.emplace(ad, gh);
                            else if (ad < sq_top.top().first) { sq_top.pop(); sq_top.emplace(ad, gh); }
                        }
                    }
                }
            }
        }

        using ExactPair = std::pair<float, uint64_t>;
        std::priority_queue<ExactPair> exact_top;
        while (!sq_top.empty()) {
            uint64_t gh = sq_top.top().second;
            sq_top.pop();
            uint32_t s_i = static_cast<uint32_t>(gh >> 32);
            uint32_t lid = static_cast<uint32_t>(gh & 0xFFFFFFFFu);
            const ScanSeg& seg = segs_[s_i];
            float ed = simd::l2_sqr_fp32(q_vec, seg.fp32_vecs.data + static_cast<std::size_t>(lid) * dim_, dim_);
            out.stats.distance_computations++;
            if (exact_top.size() < k) exact_top.emplace(ed, seg.ext_ids[lid]);
            else if (ed < exact_top.top().first) { exact_top.pop(); exact_top.emplace(ed, seg.ext_ids[lid]); }
        }
        out.ids.resize(exact_top.size());
        out.distances.resize(exact_top.size());
        for (std::size_t idx = exact_top.size(); idx > 0; --idx) {
            out.ids[idx - 1] = exact_top.top().second;
            out.distances[idx - 1] = exact_top.top().first;
            exact_top.pop();
        }
        auto t1 = std::chrono::steady_clock::now();
        out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }


    std::size_t memory_bytes() const noexcept {
        std::size_t b = 0;
        for (const auto& seg : segs_) {
            b += seg.ext_ids.size() * sizeof(uint64_t) + seg.fp32_vecs.size * sizeof(float);
            for (const auto& c : seg.cells) {
                b += (c.tv_s.size + c.tv_e.size + c.tk_s.size + c.tk_e.size) * sizeof(uint32_t)
                    + c.sq8_codes.size * sizeof(int8_t)
                    + c.sq8_meta.size() * sizeof(simd::SQ8VectorMeta)
                    + c.local_ids.size() * sizeof(uint32_t);
            }
        }
        return b;
    }

private:
    struct ScanCell {
        uint32_t min_tv_s = INF_TICK, max_tv_s = 0;
        uint32_t min_tk_s = INF_TICK, max_tk_s = 0;
        uint32_t max_tv_e = 0, max_tk_e = 0;
        AlignedBuffer<uint32_t, 64> tv_s, tv_e, tk_s, tk_e;
        std::vector<uint32_t> local_ids;
        AlignedBuffer<int8_t, 64> sq8_codes;
        std::vector<simd::SQ8VectorMeta> sq8_meta;
    };

    struct ScanSeg {
        uint64_t cap = 131072;
        uint64_t ep_cap = 8192;
        uint8_t active_gk = 0;
        uint64_t gk_counts[16] = {0};
        uint32_t k_min[16] = {0};
        uint32_t k_max[16] = {0};
        uint32_t min_tv_s = INF_TICK, max_tv_s = 0;
        uint32_t min_tk_s = INF_TICK, max_tk_s = 0;
        uint32_t max_tv_e = 0, max_tk_e = 0;
        std::vector<uint32_t> v_bounds;
        std::size_t num_nodes = 0;
        std::vector<uint64_t> ext_ids;
        AlignedBuffer<float, 64> fp32_vecs;
        std::vector<ScanCell> cells;

        ScanSeg(uint32_t bins, uint64_t c) : cap(c), ep_cap(std::max<uint64_t>(64, (c + bins - 1) / bins)), cells(bins * bins) {
            for (int i = 0; i < 16; ++i) k_min[i] = INF_TICK;
        }
    };

    std::size_t dim_, padded_dim_;
    uint32_t grid_bins_;
    uint64_t initial_cap_;
    std::size_t total_nodes_ = 0;
    uint32_t hist_min_tv_ = INF_TICK;
    uint64_t hist_lag_sum_ = 0, hist_lag_cnt_ = 0;
    std::vector<ScanSeg> segs_;

    void calibrate_seg(ScanSeg& seg, const uint32_t* tv_s, const uint32_t* tk_s, const std::size_t* ord, std::size_t avail) {
        std::size_t wb = std::min<std::size_t>(avail, 2048);
        std::vector<uint32_t> vs(wb), ks(wb);
        for (std::size_t i = 0; i < wb; ++i) { vs[i] = tv_s[ord[i]]; ks[i] = tk_s[ord[i]]; }
        std::sort(vs.begin(), vs.end());
        std::sort(ks.begin(), ks.end());
        uint32_t v_lo = (segs_.size() <= 1) ? vs.front() : std::min(hist_min_tv_, vs.front());
        double dt = (wb > 1 && ks.back() > ks.front()) ? double(ks.back() - ks.front()) / double(wb) : 300.0;
        uint64_t v_hi64 = static_cast<uint64_t>(ks.front()) + static_cast<uint64_t>(std::max(86400.0 * 90.0, dt * double(seg.cap)));
        uint32_t v_hi = (v_hi64 >= INF_TICK - 1000u) ? (INF_TICK - 1000u) : static_cast<uint32_t>(std::max<uint64_t>(v_lo + grid_bins_, v_hi64));
        for (uint32_t b = 1; b < grid_bins_; ++b) {
            uint32_t tick = v_lo + static_cast<uint32_t>((double(b) / double(grid_bins_)) * double(v_hi - v_lo));
            seg.v_bounds.push_back(tick);
        }
    }

    void append_to_seg(
        ScanSeg& seg,
        const uint64_t* ext_ids, const float* vectors,
        const uint32_t* tv_s, const uint32_t* tv_e,
        const uint32_t* tk_s, const uint32_t* tk_e,
        const std::size_t* ord, std::size_t n) {
        std::size_t base = seg.num_nodes;
        seg.ext_ids.resize(base + n);
        seg.fp32_vecs.resize((base + n) * dim_);
        std::vector<std::vector<uint32_t>> bucket(grid_bins_ * grid_bins_);
        std::vector<std::size_t> lid_to_src(n);

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t src = ord[i];
            uint32_t lid = static_cast<uint32_t>(base + i);
            lid_to_src[i] = src;
            uint32_t vs = tv_s[src], ve = tv_e[src], ks = tk_s[src], ke = tk_e[src];
            if (static_cast<uint32_t>(seg.active_gk) + 1u < grid_bins_ && seg.gk_counts[seg.active_gk] >= seg.ep_cap && ks >= seg.k_max[seg.active_gk]) {
                seg.active_gk++;
            }
            uint8_t gk = std::min<uint8_t>(seg.active_gk, static_cast<uint8_t>(grid_bins_ - 1));
            uint8_t gv = static_cast<uint8_t>(std::upper_bound(seg.v_bounds.begin(), seg.v_bounds.end(), vs) - seg.v_bounds.begin());
            seg.gk_counts[gk]++;
            if (ks < seg.k_min[gk]) seg.k_min[gk] = ks;
            if (ks > seg.k_max[gk]) seg.k_max[gk] = ks;
            if (vs < seg.min_tv_s) seg.min_tv_s = vs;
            if (vs > seg.max_tv_s) seg.max_tv_s = vs;
            if (ks < seg.min_tk_s) seg.min_tk_s = ks;
            if (ks > seg.max_tk_s) seg.max_tk_s = ks;
            if (ve > seg.max_tv_e) seg.max_tv_e = ve;
            if (ke > seg.max_tk_e) seg.max_tk_e = ke;
            if (vs < hist_min_tv_) hist_min_tv_ = vs;
            if (ks >= vs) { hist_lag_sum_ += (ks - vs); hist_lag_cnt_++; }

            seg.ext_ids[lid] = ext_ids[src];
            std::memcpy(seg.fp32_vecs.data + static_cast<std::size_t>(lid) * dim_, vectors + src * dim_, dim_ * sizeof(float));
            bucket[gk * grid_bins_ + gv].push_back(lid);
        }

        for (uint32_t cid = 0; cid < seg.cells.size(); ++cid) {
            const auto& lids = bucket[cid];
            if (lids.empty()) continue;
            ScanCell& cell = seg.cells[cid];
            std::size_t old_sz = cell.local_ids.size();
            cell.tv_s.size = old_sz; cell.tv_e.size = old_sz;
            cell.tk_s.size = old_sz; cell.tk_e.size = old_sz;
            for (uint32_t lid : lids) {
                std::size_t src = lid_to_src[lid - base];
                uint32_t slot = static_cast<uint32_t>(cell.local_ids.size());
                cell.local_ids.push_back(lid);
                uint32_t vs = tv_s[src], ve = tv_e[src], ks = tk_s[src], ke = tk_e[src];
                cell.tv_s.push_back(vs); cell.tv_e.push_back(ve);
                cell.tk_s.push_back(ks); cell.tk_e.push_back(ke);
                if (vs < cell.min_tv_s) cell.min_tv_s = vs;
                if (vs > cell.max_tv_s) cell.max_tv_s = vs;
                if (ks < cell.min_tk_s) cell.min_tk_s = ks;
                if (ks > cell.max_tk_s) cell.max_tk_s = ks;
                if (ve > cell.max_tv_e) cell.max_tv_e = ve;
                if (ke > cell.max_tk_e) cell.max_tk_e = ke;
                std::size_t off = static_cast<std::size_t>(slot) * padded_dim_;
                cell.sq8_codes.resize(off + padded_dim_, 0);
                cell.sq8_meta.push_back(simd::quantize_db_sq8(
                    seg.fp32_vecs.data + static_cast<std::size_t>(lid) * dim_,
                    cell.sq8_codes.data + off, dim_));
            }
            while ((cell.tv_s.size & 7u) != 0) {
                cell.tv_s.push_back(INF_TICK); cell.tv_e.push_back(0u);
                cell.tk_s.push_back(INF_TICK); cell.tk_e.push_back(0u);
            }
        }
        seg.num_nodes += n;
        total_nodes_ += n;
    }
};

} // namespace pitmem
