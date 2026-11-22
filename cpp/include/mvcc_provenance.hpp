#pragma once

#include "pit_types.hpp"
#include "bit_hnsw.hpp"
#include "rap_compact.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <chrono>
#include <shared_mutex>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include <algorithm>
#include <fstream>
#include <cstdio>

namespace pitmem {

struct ProvenanceRecord {
    uint64_t id = 0;
    MemoryTier tier = MemoryTier::RAW;
    uint32_t tv_s = 0;
    uint32_t tv_e = INF_TICK;
    uint32_t tk_s = 0;
    uint32_t tk_e = INF_TICK;

    // Upstream causal parents (evidence nodes from which a REFLECT node was synthesized)
    std::vector<uint64_t> parents;
    // Downstream dependents (reverse DAG edges for microsecond RESTATE taint propagation)
    std::vector<uint64_t> children;

    // Lazy-Branch MVCC state
    uint32_t suspect_after_tk = INF_TICK;
    uint32_t repaired_at_tk = INF_TICK;
    // Pairs of (tv_chg, tk_evolve) from upstream EVOLVE splits overlapping this reflection's valid window
    std::vector<std::pair<uint32_t, uint32_t>> evolve_watermarks;
    std::shared_ptr<std::atomic<uint32_t>> read_hotness_tainted = std::make_shared<std::atomic<uint32_t>>(0);
    // Atomic repair state: 0 = unqueued, 1 = queued via lock-free CAS, 2 = branch materialized
    std::shared_ptr<std::atomic<uint8_t>> repair_state = std::make_shared<std::atomic<uint8_t>>(0);
    // Maps restated upstream old_id -> replacement new_id
    std::vector<std::pair<uint64_t, uint64_t>> replacements;

    inline bool is_tainted_at(uint32_t tv_q, uint32_t tk_q) const noexcept {
        if (tier != MemoryTier::REFLECT) return false;
        if (tk_q >= suspect_after_tk) return true;
        for (const auto& ew : evolve_watermarks) {
            if (tv_q >= ew.first && tk_q >= ew.second) return true;
        }
        return false;
    }

    inline bool is_tainted_on_window(uint32_t win_tvs, uint32_t win_tve, uint32_t tk_q) const noexcept {
        (void)win_tvs;
        if (tier != MemoryTier::REFLECT) return false;
        if (tk_q >= suspect_after_tk) return true;
        for (const auto& ew : evolve_watermarks) {
            if (win_tve > ew.first && tk_q >= ew.second) return true;
        }
        return false;
    }
};

struct EvolveSplitInfo {
    uint64_t hist_id = 0;
    uint64_t new_id = 0;
    uint32_t tv_chg = 0;
    uint32_t tk_evolve = 0;
};

struct TxCommitStats {
    double latency_us = 0.0;
    uint32_t tainted_reflections = 0;
    uint32_t dag_edges_traversed = 0;
    bool split_historical_slice = false;
    bool committed = false;
    uint64_t hist_slice_id = 0;
};

struct MVCCReadResult {
    std::vector<uint64_t> resolved_ids;               // Final look-ahead-free exact Top-K context IDs (|O| <= k)
    std::vector<float> distances;                     // Exact FP32 squared L2 distances after post-unpack re-ranking
    std::vector<uint64_t> unpacked_dirty_reflections; // Tainted reflections intercepted on read
    std::vector<uint64_t> hot_repair_candidates;      // Tainted reflections atomically claimed via CAS for repair
    uint32_t unpacked_ancestors_evaluated = 0;        // Clean ancestors inspected & re-ranked during unpacking
    uint32_t raw_items_returned = 0;                  // Count of RAW leaf records in final Top-K
    uint32_t reflect_items_returned = 0;              // Count of clean REFLECT summaries in final Top-K
    uint32_t refill_rounds = 0;                       // Adaptive k_fetch doubling rounds triggered under clustered bursts
    SearchStats index_stats;
    double total_latency_us = 0.0;
};

struct TieredPitResult {
    std::vector<uint64_t> resolved_ids;               // Merged Top-K IDs across {RAW, REFLECT, COMPACT}
    std::vector<float> distances;                     // Unified unit-hypersphere squared-L2 distances in [0, 4]
    std::vector<uint8_t> item_tiers;                  // 0 = RAW, 1 = REFLECT, 2 = COMPACT
    std::vector<std::string> cold_summaries;          // Snippet for COMPACT items (empty for hot items)
    std::vector<RiskAnchorInput> matched_anchors;     // Lossless 56-B PiT-visible typed anchors joined from cold tier
    uint32_t hot_items_returned = 0;
    uint32_t cold_items_returned = 0;
    uint32_t refill_rounds = 0;
    double hot_latency_us = 0.0;
    double cold_latency_us = 0.0;
    double total_latency_us = 0.0;
};

struct ConcurrentMixedStats {
    double elapsed_sec = 0.0;
    uint64_t total_reads = 0;
    double reader_qps = 0.0;
    double reader_p50_us = 0.0;
    double reader_p95_us = 0.0;
    double reader_p99_us = 0.0;
    uint64_t total_writes = 0;
    uint64_t attempted_writes = 0;
    uint64_t rejected_writes = 0;
    double writer_ops_per_s = 0.0;
    double writer_p50_us = 0.0;
    double writer_p99_us = 0.0;
    uint32_t rollovers_triggered = 0;
    uint64_t tainted_reflections = 0;
    uint64_t cardinality_violations = 0;
};

// Phase-Fair Cohort Reader-Writer Latch (Brandenburg & Anderson):
// Eliminates Linux glibc PTHREAD_RWLOCK_PREFER_READER_NP writer starvation under continuous multi-threaded
// snapshot reads while releasing entire concurrent reader cohorts per phase to prevent reader starvation.
class PhaseFairSharedMutex {
public:
    void lock_shared() {
        std::unique_lock<std::mutex> lk(mu_);
        if (!writer_active_ && waiting_writers_ == 0) {
            ++active_readers_;
            return;
        }
        ++waiting_readers_;
        const uint64_t wait_phase = reader_phase_;
        cv_readers_.wait(lk, [&]() { return reader_phase_ != wait_phase; });
    }

    void unlock_shared() {
        std::unique_lock<std::mutex> lk(mu_);
        if (--active_readers_ == 0 && waiting_writers_ > 0) {
            writer_turn_ = true;
            cv_writers_.notify_one();
        }
    }

    void lock() {
        std::unique_lock<std::mutex> lk(mu_);
        ++waiting_writers_;
        cv_writers_.wait(lk, [&]() {
            return !writer_active_ && active_readers_ == 0 && (writer_turn_ || waiting_readers_ == 0);
        });
        --waiting_writers_;
        writer_turn_ = false;
        writer_active_ = true;
    }

    void unlock() {
        std::unique_lock<std::mutex> lk(mu_);
        writer_active_ = false;
        if (waiting_readers_ > 0) {
            active_readers_ += waiting_readers_;
            waiting_readers_ = 0;
            ++reader_phase_;
            cv_readers_.notify_all();
        } else if (waiting_writers_ > 0) {
            writer_turn_ = true;
            cv_writers_.notify_one();
        }
    }

private:
    std::mutex mu_;
    std::condition_variable cv_readers_;
    std::condition_variable cv_writers_;
    uint32_t active_readers_ = 0;
    uint32_t waiting_readers_ = 0;
    uint32_t waiting_writers_ = 0;
    uint64_t reader_phase_ = 0;
    bool writer_active_ = false;
    bool writer_turn_ = false;
};

class LazyBranchMVCCEngine {
public:
    static constexpr uint64_t HIST_SLICE_BASE = (1ULL << 62);

    explicit LazyBranchMVCCEngine(const BiTHNSWConfig& idx_cfg, uint32_t hotness_thresh = 3)
        : index_(idx_cfg), dim_(idx_cfg.dim), hotness_thresh_(hotness_thresh) {}

    ~LazyBranchMVCCEngine() {
        close_wal();
    }

    void calibrate_bounds(const uint32_t* tv_s, const uint32_t* tk_s, std::size_t n) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        index_.calibrate_bounds(tv_s, tk_s, n);
    }

    // Bulk load RAW or REFLECT records
    void bulk_load(
        const uint64_t* ids,
        const float* vectors,
        const uint32_t* tv_s,
        const uint32_t* tv_e,
        const uint32_t* tk_s,
        const uint32_t* tk_e,
        const uint8_t* tiers,
        std::size_t n,
        int num_threads = 0) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        index_.build_batch(ids, vectors, tv_s, tv_e, tk_s, tk_e, n, num_threads);
        records_.reserve(records_.size() + n);
        for (std::size_t i = 0; i < n; ++i) {
            ProvenanceRecord rec;
            rec.id = ids[i];
            rec.tier = static_cast<MemoryTier>(tiers ? tiers[i] : 0);
            rec.tv_s = tv_s[i];
            rec.tv_e = tv_e[i];
            rec.tk_s = tk_s[i];
            rec.tk_e = tk_e[i];
            records_[ids[i]] = std::move(rec);
            if (ids[i] >= HIST_SLICE_BASE && ids[i] >= next_auto_id_) next_auto_id_ = ids[i] + 1;
        }
    }

    void attach_cold_store(const RAPCompactStore* cold_store) noexcept {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        cold_store_ = cold_store;
    }

    bool insert_raw(
        uint64_t id,
        const float* vec,
        uint32_t tv_s, uint32_t tv_e,
        uint32_t tk_s, uint32_t tk_e) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        if (records_.find(id) != records_.end() || tv_s >= tv_e || tk_s >= tk_e) return false;
        begin_engine_wal_tx();
        index_.insert_single(id, vec, tv_s, tv_e, tk_s, tk_e);
        ProvenanceRecord rec;
        rec.id = id;
        rec.tier = MemoryTier::RAW;
        rec.tv_s = tv_s;
        rec.tv_e = tv_e;
        rec.tk_s = tk_s;
        rec.tk_e = tk_e;
        records_[id] = std::move(rec);
        if (id >= HIST_SLICE_BASE && id >= next_auto_id_) next_auto_id_ = id + 1;
        append_prov_wal_op(1, id, 0, 0, tv_s, tv_e, tk_s, tk_e, vec, nullptr, 0);
        commit_engine_wal_tx();
        return true;
    }

    // Register causal provenance edges: reflection_id depends on parent_ids
    void register_provenance(uint64_t reflection_id, const std::vector<uint64_t>& parent_ids) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        auto it = records_.find(reflection_id);
        if (it == records_.end()) return;
        begin_engine_wal_tx();
        it->second.parents = parent_ids;
        bind_parents_and_inherit_taint(it->second);
        append_prov_wal_op(2, reflection_id, 0, 0, 0, 0, 0, 0, nullptr, parent_ids.data(), parent_ids.size());
        commit_engine_wal_tx();
    }

    // Insert a newly synthesized REFLECT node with causal lineage
    void insert_reflection(
        uint64_t reflection_id,
        const float* vec,
        uint32_t tv_s, uint32_t tv_e,
        uint32_t tk_s, uint32_t tk_e,
        const std::vector<uint64_t>& parent_ids) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        if (records_.find(reflection_id) != records_.end() || tv_s >= tv_e || tk_s >= tk_e) return;
        begin_engine_wal_tx();
        insert_reflection_unlocked(reflection_id, vec, tv_s, tv_e, tk_s, tk_e, parent_ids);
        const auto& r = records_[reflection_id];
        append_prov_wal_op(3, reflection_id, 0, 0, r.tv_s, r.tv_e, r.tk_s, r.tk_e, vec, parent_ids.data(), parent_ids.size());
        commit_engine_wal_tx();
    }

    // Operator 1: EVOLVE (Natural state evolution at tv_chg learned at tk_now)
    // Preserves pre-tk_now belief AND historical [tv_s, tv_chg) truth without tainting prior reflections!
    TxCommitStats apply_evolve(
        uint64_t old_id,
        uint64_t new_id,
        const float* new_vec,
        uint32_t tv_chg,
        uint32_t new_tv_e,
        uint32_t tk_now) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        auto t0 = std::chrono::steady_clock::now();
        TxCommitStats st;
        (void)new_tv_e;

        // Enforce strict operator preconditions before any mutation (Proposition 1)
        if (new_id == old_id || records_.find(new_id) != records_.end()) return st;
        auto it = records_.find(old_id);
        if (it == records_.end()) return st;
        if (it->second.tk_e != INF_TICK || it->second.tk_s >= tk_now) return st;
        if (!(it->second.tv_s < tv_chg && tv_chg < it->second.tv_e)) return st;

        const uint32_t old_tvs = it->second.tv_s;
        const uint32_t old_tve = it->second.tv_e;
        const MemoryTier old_tier = it->second.tier;
        std::vector<uint64_t> old_parents = it->second.parents;
        std::vector<uint64_t> old_children = it->second.children;

        // Allocate historical continuation ID strictly inside reserved HIST_SLICE_BASE namespace (zero collision with caller IDs!)
        if (next_auto_id_ < HIST_SLICE_BASE) next_auto_id_ = HIST_SLICE_BASE;
        uint64_t hist_slice_id = next_auto_id_++;
        while (records_.find(hist_slice_id) != records_.end() || hist_slice_id == new_id) {
            hist_slice_id = next_auto_id_++;
        }

        begin_engine_wal_tx();
        // 1) Close old version's tk_e = tk_now so queries with tk_q < tk_now see unclipped old_tve
        index_.update_tx_end(old_id, tk_now);
        it->second.tk_e = tk_now;
        cold_retracted_tk_[old_id] = tk_now;

        // 2) Insert historical truth continuation [old_tvs, tv_chg) x [tk_now, +inf)
        const float* old_vp = index_.get_vector_ptr(old_id);
        std::vector<float> old_v_copy(dim_, 0.0f);
        if (old_vp != nullptr) {
            std::memcpy(old_v_copy.data(), old_vp, dim_ * sizeof(float));
        }
        index_.insert_single(hist_slice_id, old_v_copy.data(), old_tvs, tv_chg, tk_now, INF_TICK);
        ProvenanceRecord hrec;
        hrec.id = hist_slice_id;
        hrec.tier = old_tier;
        hrec.tv_s = old_tvs;
        hrec.tv_e = tv_chg;
        hrec.tk_s = tk_now;
        hrec.tk_e = INF_TICK;
        hrec.parents = old_parents;
        for (uint64_t cid : old_children) {
            auto cit = records_.find(cid);
            if (cit != records_.end() && cit->second.tv_s < tv_chg && old_tvs < cit->second.tv_e) {
                add_unique_child(hrec.children, cid);
            }
        }
        records_[hist_slice_id] = std::move(hrec);
        bind_parents_and_inherit_taint(records_[hist_slice_id]);
        st.split_historical_slice = true;
        evolve_slices_[old_id].push_back(EvolveSplitInfo{hist_slice_id, new_id, tv_chg, tk_now});

        // 3) Insert evolved slice [tv_chg, old_tve) x [tk_now, +inf) inheriting old_tve
        index_.insert_single(new_id, new_vec, tv_chg, old_tve, tk_now, INF_TICK);
        ProvenanceRecord nrec;
        nrec.id = new_id;
        nrec.tier = old_tier;
        nrec.tv_s = tv_chg;
        nrec.tv_e = old_tve;
        nrec.tk_s = tk_now;
        nrec.tk_e = INF_TICK;
        nrec.parents = old_parents;
        for (uint64_t cid : old_children) {
            auto cit = records_.find(cid);
            if (cit != records_.end() && cit->second.tv_s < old_tve && tv_chg < cit->second.tv_e) {
                add_unique_child(nrec.children, cid);
            }
        }
        records_[new_id] = std::move(nrec);
        bind_parents_and_inherit_taint(records_[new_id]);

        if (!old_children.empty()) {
            // 4) Unconditional reverse-DAG BFS: mark reflections overlapping [tv_chg, old_tve)
            std::queue<uint64_t> e_bfs;
            std::unordered_set<uint64_t> e_seen;
            for (uint64_t cid : old_children) {
                if (e_seen.insert(cid).second) e_bfs.push(cid);
                st.dag_edges_traversed++;
            }
            while (!e_bfs.empty()) {
                uint64_t cid = e_bfs.front();
                e_bfs.pop();
                auto cit = records_.find(cid);
                if (cit == records_.end()) continue;
                auto& crec = cit->second;
                if (crec.tier == MemoryTier::REFLECT && crec.tk_e > tk_now &&
                    crec.tv_e > tv_chg && crec.tv_s < old_tve) {
                    add_unique_watermark(crec.evolve_watermarks, tv_chg, tk_now);
                    add_unique_child(records_[new_id].children, cid);
                    st.tainted_reflections++;
                }
                for (uint64_t gc : crec.children) {
                    st.dag_edges_traversed++;
                    if (e_seen.insert(gc).second) e_bfs.push(gc);
                }
            }
        }
        append_prov_wal_op(4, old_id, new_id, hist_slice_id, tv_chg, old_tve, tk_now, INF_TICK, new_vec, nullptr, 0);
        commit_engine_wal_tx();

        auto t1 = std::chrono::steady_clock::now();
        st.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        st.committed = true;
        st.hist_slice_id = hist_slice_id;
        return st;
    }

    // Operator 2: RESTATE (Accounting restatement / macro revision learned at tk_now)
    // Supports both hot BiT-HNSW assertions and compacted cold RAP-Compact members!
    TxCommitStats apply_restate(
        uint64_t old_id,
        uint64_t new_id,
        const float* corrected_vec,
        uint32_t tk_now) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        auto t0 = std::chrono::steady_clock::now();
        TxCommitStats st;

        if (new_id == old_id || records_.find(new_id) != records_.end()) return st;

        uint32_t orig_tvs = 0, orig_tve = INF_TICK, orig_tks = 0, orig_tke = INF_TICK;
        MemoryTier orig_tier = MemoryTier::RAW;
        std::vector<uint64_t> inherited_children;
        bool in_hot = false;

        auto it = records_.find(old_id);
        if (it != records_.end()) {
            if (it->second.tk_e != INF_TICK || it->second.tk_s >= tk_now) return st;
            orig_tvs = it->second.tv_s;
            orig_tve = it->second.tv_e;
            orig_tks = it->second.tk_s;
            orig_tke = it->second.tk_e;
            orig_tier = it->second.tier;
            inherited_children = it->second.children;
            in_hot = true;
        } else if (cold_store_ != nullptr &&
                   cold_retracted_tk_.find(old_id) == cold_retracted_tk_.end() &&
                   cold_store_->find_member_bounds(old_id, orig_tvs, orig_tve, orig_tks, orig_tke)) {
            if (orig_tke != INF_TICK || orig_tks >= tk_now) return st;
        } else {
            return st;
        }

        begin_engine_wal_tx();
        // 1) Close old version's transaction right-endpoint in O(1) across hot SoA and cold retraction overlay
        if (in_hot) {
            index_.update_tx_end(old_id, tk_now);
            it->second.tk_e = tk_now;
        }
        cold_retracted_tk_[old_id] = tk_now;
        succ_map_[old_id] = {new_id, tk_now};

        // 2) Insert restated version with IDENTICAL valid interval [orig_tvs, orig_tve) and tk = [tk_now, +inf)
        index_.insert_single(new_id, corrected_vec, orig_tvs, orig_tve, tk_now, INF_TICK);
        ProvenanceRecord nrec;
        nrec.id = new_id;
        nrec.tier = orig_tier;
        nrec.tv_s = orig_tvs;
        nrec.tv_e = orig_tve;
        nrec.tk_s = tk_now;
        nrec.tk_e = INF_TICK;
        nrec.children = inherited_children;
        records_[new_id] = std::move(nrec);
        if (new_id >= HIST_SLICE_BASE && new_id >= next_auto_id_) next_auto_id_ = new_id + 1;

        // 3) Zero-LLM Causal Lineage Taint Propagation (unconditional reverse-DAG BFS across all intermediates!)
        std::queue<uint64_t> bfs;
        std::unordered_set<uint64_t> seen;
        for (uint64_t child_id : inherited_children) {
            if (seen.insert(child_id).second) {
                bfs.push(child_id);
            }
            st.dag_edges_traversed++;
        }

        while (!bfs.empty()) {
            uint64_t cid = bfs.front();
            bfs.pop();
            auto cit = records_.find(cid);
            if (cit == records_.end()) continue;
            auto& crec = cit->second;

            bool valid_overlap = (crec.tv_s < orig_tve) && (orig_tvs < crec.tv_e);
            if (crec.tk_e > tk_now && valid_overlap) {
                if (tk_now < crec.suspect_after_tk) {
                    crec.suspect_after_tk = tk_now;
                }
                crec.replacements.emplace_back(old_id, new_id);
                add_unique_child(records_[new_id].children, cid);
                st.tainted_reflections++;
            }
            for (uint64_t gchild : crec.children) {
                st.dag_edges_traversed++;
                if (seen.insert(gchild).second) {
                    bfs.push(gchild);
                }
            }
        }
        append_prov_wal_op(5, old_id, new_id, in_hot ? 1ULL : 0ULL, orig_tvs, orig_tve, tk_now, INF_TICK, corrected_vec, nullptr, 0);
        commit_engine_wal_tx();

        auto t1 = std::chrono::steady_clock::now();
        st.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        st.committed = true;
        return st;
    }

    // Materialize a repaired reflection branch at actual materialization time branch_tk:
    // - Never backdates branch_tk prior to any clean parent's birth timestamp (preserving LCR = 0!).
    // - Recursively resolves multi-hop tainted ancestors to causally clean versions at branch_tk (preserving STR = 0!).
    // - Keeps dirty_reflection_id's suspect_after_tk active on [suspect_after_tk, branch_tk) so historical queries
    //   in the intermediate window before materialization deterministically unpack to snapshot-valid leaves!
    void materialize_repaired_reflection(
        uint64_t dirty_reflection_id,
        uint64_t new_reflection_id,
        const float* new_reflection_vec,
        uint32_t branch_tk) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        if (new_reflection_id == dirty_reflection_id || records_.find(new_reflection_id) != records_.end()) return;
        auto it = records_.find(dirty_reflection_id);
        if (it == records_.end() || it->second.repair_state->load(std::memory_order_acquire) == 2) return;

        uint32_t tvs = it->second.tv_s;
        uint32_t tve = it->second.tv_e;
        std::vector<uint64_t> inherited_children = it->second.children;

        // Recursively collect causally clean ancestors valid at branch_tk and ensure branch_tk >= max parent tk_s
        std::vector<uint64_t> clean_parents;
        std::unordered_set<uint64_t> p_seen;
        uint32_t min_valid_mat_tk = branch_tk;
        if (it->second.suspect_after_tk < INF_TICK) {
            min_valid_mat_tk = std::max(min_valid_mat_tk, it->second.suspect_after_tk);
        }
        for (const auto& ew : it->second.evolve_watermarks) {
            if (ew.first < tve && ew.second < INF_TICK) {
                min_valid_mat_tk = std::max(min_valid_mat_tk, ew.second);
            }
        }
        collect_clean_parents_for_repair(dirty_reflection_id, tvs, tve, min_valid_mat_tk, p_seen, clean_parents, min_valid_mat_tk);

        begin_engine_wal_tx();
        // Close dirty reflection at min_valid_mat_tk; queries in [suspect_after_tk, min_valid_mat_tk) still unpack cleanly!
        index_.update_tx_end(dirty_reflection_id, min_valid_mat_tk);
        it->second.tk_e = min_valid_mat_tk;
        it->second.repaired_at_tk = min_valid_mat_tk;
        it->second.repair_state->store(2, std::memory_order_release);
        succ_map_[dirty_reflection_id] = {new_reflection_id, min_valid_mat_tk};

        insert_reflection_unlocked(
            new_reflection_id, new_reflection_vec, tvs, tve, min_valid_mat_tk, INF_TICK, clean_parents);

        // Inherit downstream meta-reflections so if dirty_reflection_id had children, they also know new_reflection_id
        for (uint64_t child_id : inherited_children) {
            add_unique_child(records_[new_reflection_id].children, child_id);
            auto cit = records_.find(child_id);
            if (cit != records_.end()) {
                cit->second.replacements.emplace_back(dirty_reflection_id, new_reflection_id);
            }
        }
        append_prov_wal_op(
            6, dirty_reflection_id, new_reflection_id, 0,
            tvs, tve, min_valid_mat_tk, INF_TICK,
            new_reflection_vec, clean_parents.data(), clean_parents.size());
        commit_engine_wal_tx();
    }

    // Copy-on-Read Snapshot-Isolated Point-in-Time Query with Adaptive Iterative k_fetch Doubling
    // and Exact Post-Unpack Top-K Re-Ranking (Frontier-Certified Clean Completeness, Thread-Safe Shared Reader Lock)
    MVCCReadResult query_pit(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t ef_search = 64,
        bool adaptive_fetch = true) {
        std::shared_lock<PhaseFairSharedMutex> lk(rw_mu_);
        auto t0 = std::chrono::steady_clock::now();
        MVCCReadResult res;
        if (k == 0) return res;

        uint32_t cur_k_fetch = std::max<uint32_t>(k * 2, k + 12);
        const uint32_t k_max = std::max<uint32_t>(k * 32, 512u);

        std::unordered_set<uint64_t> processed_raw_ids;
        std::unordered_set<uint64_t> seen_cands;
        std::unordered_set<uint64_t> raw_topk_dirty;
        std::unordered_map<uint64_t, std::vector<uint64_t>> leaf_to_dirty_refl;
        std::vector<std::pair<float, uint64_t>> scored_pool;
        processed_raw_ids.reserve(cur_k_fetch * 2);
        seen_cands.reserve(cur_k_fetch * 2);
        scored_pool.reserve(cur_k_fetch * 2);

        uint32_t round_idx = 0;
        while (true) {
            const uint32_t ef_eff = std::max<uint32_t>(ef_search, std::min<uint32_t>(cur_k_fetch * 2, 1536u));
            QueryResult raw_res = index_.search(q_vec, tv_q, tk_q, cur_k_fetch, ef_eff);
            res.index_stats.visited_cells += raw_res.stats.visited_cells;
            res.index_stats.graph_cells += raw_res.stats.graph_cells;
            res.index_stats.scan_cells += raw_res.stats.scan_cells;
            res.index_stats.distance_computations += raw_res.stats.distance_computations;
            res.index_stats.predicate_checks += raw_res.stats.predicate_checks;

            bool any_taint_in_round = false;
            for (std::size_t i = 0; i < raw_res.ids.size(); ++i) {
                uint64_t mid = raw_res.ids[i];
                if (!processed_raw_ids.insert(mid).second) continue;
                float dist = raw_res.distances[i];
                auto it = records_.find(mid);
                if (it == records_.end()) {
                    if (seen_cands.insert(mid).second) {
                        scored_pool.emplace_back(dist, mid);
                    }
                    continue;
                }
                const auto& rec = it->second;
                // Intercept ANY REFLECT node queried AT OR AFTER its restatement or evolve taint watermark!
                if (rec.is_tainted_at(tv_q, tk_q)) {
                    any_taint_in_round = true;
                    if (round_idx == 0 && i < k) {
                        raw_topk_dirty.insert(mid);
                    }
                    std::unordered_set<uint64_t> unpack_vis;
                    std::vector<uint64_t> unpacked_clean_ids;
                    unpack_clean_evidence(mid, tv_q, tk_q, seen_cands, unpack_vis, unpacked_clean_ids);
                    for (uint64_t u_id : unpacked_clean_ids) {
                        leaf_to_dirty_refl[u_id].push_back(mid);
                        const float* uv = lookup_record_vec(u_id);
                        float exact_d = uv ? simd::l2_sqr_fp32(q_vec, uv, dim_) : dist;
                        res.index_stats.distance_computations++;
                        res.unpacked_ancestors_evaluated++;
                        scored_pool.emplace_back(exact_d, u_id);
                    }
                } else {
                    if (seen_cands.insert(mid).second) {
                        scored_pool.emplace_back(dist, mid);
                    }
                }
            }

            // Frontier-Certified Clean Completeness Check:
            // Stop if (a) adaptive_fetch disabled or no taint encountered, (b) live index exhausted (< cur_k_fetch items),
            // (c) cur_k_fetch reached k_max, or (d) >= k clean/unpacked items already satisfy d <= d_frontier!
            if (!adaptive_fetch || !any_taint_in_round || raw_res.ids.size() < cur_k_fetch || cur_k_fetch >= k_max) {
                break;
            }
            const float d_frontier = raw_res.distances.back() + 1e-6f;
            std::size_t certified_clean = 0;
            for (const auto& [cd, cid] : scored_pool) {
                (void)cid;
                if (cd <= d_frontier) {
                    if (++certified_clean >= k) break;
                }
            }
            if (certified_clean >= k) {
                break;
            }
            cur_k_fetch = std::min<uint32_t>(k_max, cur_k_fetch * 2);
            res.refill_rounds++;
            round_idx++;
        }

        // Exact Top-K selection over merged clean + unpacked candidates (strictly guarantees |O| <= k!)
        std::unordered_set<uint64_t> impacted_dirty_seen;
        std::vector<uint64_t> impacted_dirty_order;
        for (uint64_t d_id : raw_topk_dirty) {
            if (impacted_dirty_seen.insert(d_id).second) {
                impacted_dirty_order.push_back(d_id);
            }
        }
        const std::size_t out_cnt = std::min<std::size_t>(k, scored_pool.size());
        if (out_cnt > 0) {
            std::partial_sort(
                scored_pool.begin(),
                scored_pool.begin() + static_cast<std::ptrdiff_t>(out_cnt),
                scored_pool.end());
            res.resolved_ids.resize(out_cnt);
            res.distances.resize(out_cnt);
            for (std::size_t j = 0; j < out_cnt; ++j) {
                uint64_t cid = scored_pool[j].second;
                res.resolved_ids[j] = cid;
                res.distances[j] = scored_pool[j].first;
                auto l_it = leaf_to_dirty_refl.find(cid);
                if (l_it != leaf_to_dirty_refl.end()) {
                    for (uint64_t d_id : l_it->second) {
                        if (impacted_dirty_seen.insert(d_id).second) {
                            impacted_dirty_order.push_back(d_id);
                        }
                    }
                }
                auto rit = records_.find(cid);
                if (rit != records_.end() && rit->second.tier == MemoryTier::REFLECT) {
                    res.reflect_items_returned++;
                } else {
                    res.raw_items_returned++;
                }
            }
        }

        // Update hotness counters & CAS(0 -> 1) repair claims ONLY for tainted reflections impacting Top-K!
        for (uint64_t mid : impacted_dirty_order) {
            res.unpacked_dirty_reflections.push_back(mid);
            auto dit = records_.find(mid);
            if (dit != records_.end() && dit->second.repair_state->load(std::memory_order_acquire) == 0) {
                uint32_t cur_hot = dit->second.read_hotness_tainted->fetch_add(1, std::memory_order_relaxed) + 1;
                if (cur_hot >= hotness_thresh_) {
                    uint8_t expected = 0;
                    if (dit->second.repair_state->compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                        res.hot_repair_candidates.push_back(mid);
                    }
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        res.total_latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        res.index_stats.latency_us = res.total_latency_us;
        return res;
    }

    // Unified End-to-End Hot ({RAW, REFLECT}) + Cold ({COMPACT}) Tiered Point-in-Time Search
    // Merges hot BiT-HNSW + Lazy-Branch MVCC candidates with unit-hypersphere-calibrated cold RAP-Compact clusters
    // and joins lossless 56-byte bi-temporal typed anchors in a single call.
    TieredPitResult query_tiered_pit(
        const float* q_vec,
        const std::string& entity_filter,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t ef_search = 64) {
        auto t0 = std::chrono::steady_clock::now();
        TieredPitResult out;
        if (k == 0) return out;

        MVCCReadResult hot_res = query_pit(q_vec, tv_q, tk_q, k, ef_search, true);
        out.hot_latency_us = hot_res.total_latency_us;
        out.refill_rounds = hot_res.refill_rounds;

        const RAPCompactStore* cold_ptr = nullptr;
        std::unordered_map<uint64_t, uint32_t> cold_retracted;
        {
            std::shared_lock<PhaseFairSharedMutex> lk(rw_mu_);
            cold_ptr = cold_store_;
            cold_retracted = cold_retracted_tk_;
        }

        HybridRAPResult cold_res;
        if (cold_ptr != nullptr) {
            cold_res = cold_ptr->search_hybrid_pit(
                q_vec, entity_filter, tv_q, tk_q, k,
                cold_retracted.empty() ? nullptr : &cold_retracted);
            out.cold_latency_us = cold_res.latency_us;
            out.matched_anchors = std::move(cold_res.matched_anchors);
        }

        struct MergedCand {
            float dist;
            uint64_t id;
            uint8_t tier;
            std::string summary;
            bool operator<(const MergedCand& o) const noexcept {
                return (dist != o.dist) ? (dist < o.dist) : (id < o.id);
            }
        };

        std::vector<MergedCand> merged;
        merged.reserve(hot_res.resolved_ids.size() + cold_res.member_ids.size());
        std::unordered_set<uint64_t> seen_ids;
        seen_ids.reserve(hot_res.resolved_ids.size() + cold_res.member_ids.size());

        {
            std::shared_lock<PhaseFairSharedMutex> lk(rw_mu_);
            for (std::size_t i = 0; i < hot_res.resolved_ids.size(); ++i) {
                uint64_t hid = hot_res.resolved_ids[i];
                if (!seen_ids.insert(hid).second) continue;
                uint8_t t_code = static_cast<uint8_t>(MemoryTier::RAW);
                auto it = records_.find(hid);
                if (it != records_.end()) t_code = static_cast<uint8_t>(it->second.tier);
                merged.push_back(MergedCand{hot_res.distances[i], hid, t_code, std::string()});
            }
        }
        for (std::size_t j = 0; j < cold_res.member_ids.size(); ++j) {
            uint64_t cid = cold_res.member_ids[j];
            auto crit = cold_retracted.find(cid);
            if (crit != cold_retracted.end() && crit->second <= tk_q) continue;
            if (!seen_ids.insert(cid).second) continue;
            merged.push_back(MergedCand{
                cold_res.cluster_distances[j],
                cid,
                static_cast<uint8_t>(MemoryTier::COMPACT),
                cold_res.cluster_summaries[j]
            });
        }

        const std::size_t keep = std::min<std::size_t>(k, merged.size());
        if (keep > 0) {
            std::partial_sort(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(keep), merged.end());
            out.resolved_ids.resize(keep);
            out.distances.resize(keep);
            out.item_tiers.resize(keep);
            out.cold_summaries.resize(keep);
            std::unordered_set<uint64_t> kept_set;
            kept_set.reserve(keep);
            for (std::size_t i = 0; i < keep; ++i) {
                out.resolved_ids[i] = merged[i].id;
                out.distances[i] = merged[i].dist;
                out.item_tiers[i] = merged[i].tier;
                out.cold_summaries[i] = std::move(merged[i].summary);
                kept_set.insert(merged[i].id);
                if (merged[i].tier == static_cast<uint8_t>(MemoryTier::COMPACT)) {
                    out.cold_items_returned++;
                } else {
                    out.hot_items_returned++;
                }
            }
            if (entity_filter.empty() && !out.matched_anchors.empty()) {
                std::vector<RiskAnchorInput> filtered_anc;
                filtered_anc.reserve(out.matched_anchors.size());
                for (auto& a : out.matched_anchors) {
                    if (kept_set.count(a.source_mem_id)) filtered_anc.push_back(std::move(a));
                }
                out.matched_anchors = std::move(filtered_anc);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        out.total_latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }

    // Native C++17 Multi-Threaded Mixed Read/Write/Restate/Rollover Concurrency Stress Harness
    ConcurrentMixedStats benchmark_concurrent_mixed(
        const float* query_vecs,
        const uint32_t* tv_queries,
        const uint32_t* tk_queries,
        std::size_t num_queries,
        const uint64_t* victim_ids,
        const float* write_vecs,
        std::size_t num_write_ops,
        uint32_t base_tk_now,
        double restate_ratio,
        double insert_ratio,
        int num_reader_threads,
        uint32_t k = 10,
        uint32_t ef_search = 64,
        double duration_sec = 2.0) {
        ConcurrentMixedStats out;
        if (num_queries == 0 || num_reader_threads <= 0) return out;

        uint32_t segs_before = 0;
        {
            std::shared_lock<PhaseFairSharedMutex> lk(rw_mu_);
            segs_before = index_.num_segments();
        }

        std::atomic<bool> start_flag{false};
        std::atomic<bool> stop_flag{false};
        std::atomic<uint64_t> latest_tk_published{base_tk_now};
        std::atomic<uint64_t> card_violations{0};
        std::atomic<uint64_t> tainted_sum{0};

        std::vector<std::vector<double>> reader_lats(static_cast<std::size_t>(num_reader_threads));
        for (auto& v : reader_lats) v.reserve(32768);
        std::vector<double> writer_lats;
        writer_lats.reserve(num_write_ops > 0 ? std::max<std::size_t>(num_write_ops, 16384) : 16);

        std::vector<std::thread> r_threads;
        r_threads.reserve(static_cast<std::size_t>(num_reader_threads));
        for (int tid = 0; tid < num_reader_threads; ++tid) {
            r_threads.emplace_back([&, tid]() {
                while (!start_flag.load(std::memory_order_acquire)) {}
                std::size_t q_cursor = static_cast<std::size_t>(tid) * 17u;
                auto& my_lats = reader_lats[static_cast<std::size_t>(tid)];
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    std::size_t qi = q_cursor % num_queries;
                    q_cursor++;
                    const float* qv = query_vecs + qi * dim_;
                    uint32_t tvq = tv_queries[qi];
                    // Mix historical snapshot queries with live head-of-stream queries observing fresh commits
                    uint32_t tkq = (q_cursor & 1u)
                        ? tk_queries[qi]
                        : static_cast<uint32_t>(latest_tk_published.load(std::memory_order_relaxed));

                    auto rq0 = std::chrono::steady_clock::now();
                    MVCCReadResult rd = this->query_pit(qv, tvq, tkq, k, ef_search, true);
                    auto rq1 = std::chrono::steady_clock::now();
                    bool violated = (rd.resolved_ids.size() > k);
                    if (!violated && !rd.resolved_ids.empty()) {
                        std::shared_lock<PhaseFairSharedMutex> vlk(rw_mu_);
                        for (uint64_t cid : rd.resolved_ids) {
                            auto vit = records_.find(cid);
                            if (vit != records_.end()) {
                                const auto& vrec = vit->second;
                                if (!bitemp_visible(vrec.tv_s, vrec.tv_e, vrec.tk_s, vrec.tk_e, tvq, tkq) ||
                                    vrec.is_tainted_at(tvq, tkq)) {
                                    violated = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (violated) {
                        card_violations.fetch_add(1, std::memory_order_relaxed);
                    }
                    my_lats.push_back(std::chrono::duration<double, std::micro>(rq1 - rq0).count());
                }
            });
        }

        std::thread w_thread;
        std::atomic<uint64_t> attempted_cnt{0};
        std::atomic<uint64_t> rejected_cnt{0};
        if (num_write_ops > 0 && write_vecs != nullptr) {
            // Track active mutable lineage heads and their exact valid bounds so every EVOLVE/RESTATE satisfies preconditions
            std::vector<uint64_t> cur_vids;
            std::vector<uint32_t> cur_tvs;
            std::vector<uint32_t> cur_tve;
            if (victim_ids != nullptr) {
                std::shared_lock<PhaseFairSharedMutex> lk(rw_mu_);
                cur_vids.resize(num_write_ops, 0);
                cur_tvs.resize(num_write_ops, 1u);
                cur_tve.resize(num_write_ops, INF_TICK);
                for (std::size_t i = 0; i < num_write_ops; ++i) {
                    cur_vids[i] = victim_ids[i];
                    auto it = records_.find(victim_ids[i]);
                    if (it != records_.end()) {
                        cur_tvs[i] = it->second.tv_s;
                        cur_tve[i] = it->second.tv_e;
                    }
                }
            }

            w_thread = std::thread([&, cur_vids = std::move(cur_vids), cur_tvs = std::move(cur_tvs), cur_tve = std::move(cur_tve)]() mutable {
                while (!start_flag.load(std::memory_order_acquire)) {}
                std::size_t wi = 0;
                uint64_t new_id_gen = 920000000ULL;
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    std::size_t slot = wi % num_write_ops;
                    uint32_t tk_now = base_tk_now + static_cast<uint32_t>((wi + 1) * 5);
                    const float* wv = write_vecs + slot * dim_;
                    uint64_t nid = new_id_gen++;
                    double u = static_cast<double>((wi * 2654435761ULL) & 0xFFFFu) / 65536.0;

                    attempted_cnt.fetch_add(1, std::memory_order_relaxed);
                    bool ok_commit = false;
                    auto w0 = std::chrono::steady_clock::now();
                    if (u < insert_ratio || cur_vids.empty()) {
                        uint32_t vs = (tk_now > 7200u) ? (tk_now - 7200u) : 1u;
                        ok_commit = this->insert_raw(nid, wv, vs, INF_TICK, tk_now, INF_TICK);
                    } else if (u < insert_ratio + restate_ratio) {
                        uint64_t vid = cur_vids[slot];
                        TxCommitStats cs = this->apply_restate(vid, nid, wv, tk_now);
                        ok_commit = cs.committed;
                        if (ok_commit) {
                            cur_vids[slot] = nid;
                            tainted_sum.fetch_add(cs.tainted_reflections, std::memory_order_relaxed);
                        }
                    } else {
                        uint64_t vid = cur_vids[slot];
                        uint32_t lo_v = cur_tvs[slot];
                        uint32_t hi_v = cur_tve[slot];
                        uint32_t cap_hi = (hi_v == INF_TICK) ? std::max<uint32_t>(lo_v + 7200u, tk_now) : hi_v;
                        if (lo_v + 2u < cap_hi && cap_hi <= hi_v) {
                            uint32_t tv_chg = lo_v + std::max<uint32_t>(1u, (cap_hi - lo_v) / 4u);
                            TxCommitStats cs = this->apply_evolve(vid, nid, wv, tv_chg, hi_v, tk_now);
                            ok_commit = cs.committed;
                            if (ok_commit) {
                                cur_vids[slot] = nid;
                                cur_tvs[slot] = tv_chg;
                                tainted_sum.fetch_add(cs.tainted_reflections, std::memory_order_relaxed);
                            }
                        } else {
                            // Fallback to RESTATE if valid interval exhausted width < 2 ticks
                            TxCommitStats cs = this->apply_restate(vid, nid, wv, tk_now);
                            ok_commit = cs.committed;
                            if (ok_commit) {
                                cur_vids[slot] = nid;
                                tainted_sum.fetch_add(cs.tainted_reflections, std::memory_order_relaxed);
                            }
                        }
                    }
                    auto w1 = std::chrono::steady_clock::now();
                    if (ok_commit) {
                        latest_tk_published.store(tk_now, std::memory_order_relaxed);
                        writer_lats.push_back(std::chrono::duration<double, std::micro>(w1 - w0).count());
                    } else {
                        rejected_cnt.fetch_add(1, std::memory_order_relaxed);
                    }
                    wi++;
                    if (wi >= num_write_ops) break;
                }
            });
        }

        auto bench_t0 = std::chrono::steady_clock::now();
        start_flag.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::duration<double>(duration_sec));
        stop_flag.store(true, std::memory_order_release);

        for (auto& th : r_threads) {
            if (th.joinable()) th.join();
        }
        if (w_thread.joinable()) w_thread.join();
        auto bench_t1 = std::chrono::steady_clock::now();

        out.elapsed_sec = std::chrono::duration<double>(bench_t1 - bench_t0).count();
        std::vector<double> all_r_lats;
        std::size_t tot_r = 0;
        for (const auto& v : reader_lats) tot_r += v.size();
        all_r_lats.reserve(tot_r);
        for (const auto& v : reader_lats) all_r_lats.insert(all_r_lats.end(), v.begin(), v.end());

        out.total_reads = all_r_lats.size();
        out.reader_qps = (out.elapsed_sec > 0.0) ? (static_cast<double>(out.total_reads) / out.elapsed_sec) : 0.0;
        if (!all_r_lats.empty()) {
            std::sort(all_r_lats.begin(), all_r_lats.end());
            out.reader_p50_us = all_r_lats[std::min(all_r_lats.size() - 1, (all_r_lats.size() * 50) / 100)];
            out.reader_p95_us = all_r_lats[std::min(all_r_lats.size() - 1, (all_r_lats.size() * 95) / 100)];
            out.reader_p99_us = all_r_lats[std::min(all_r_lats.size() - 1, (all_r_lats.size() * 99) / 100)];
        }

        out.total_writes = writer_lats.size();
        out.attempted_writes = attempted_cnt.load(std::memory_order_relaxed);
        out.rejected_writes = rejected_cnt.load(std::memory_order_relaxed);
        out.writer_ops_per_s = (out.elapsed_sec > 0.0) ? (static_cast<double>(out.total_writes) / out.elapsed_sec) : 0.0;
        if (!writer_lats.empty()) {
            std::sort(writer_lats.begin(), writer_lats.end());
            out.writer_p50_us = writer_lats[std::min(writer_lats.size() - 1, (writer_lats.size() * 50) / 100)];
            out.writer_p99_us = writer_lats[std::min(writer_lats.size() - 1, (writer_lats.size() * 99) / 100)];
        }

        uint32_t segs_after = 0;
        {
            std::shared_lock<PhaseFairSharedMutex> lk(rw_mu_);
            segs_after = index_.num_segments();
        }
        out.rollovers_triggered = (segs_after >= segs_before) ? (segs_after - segs_before) : 0u;
        out.tainted_reflections = tainted_sum.load(std::memory_order_relaxed);
        out.cardinality_violations = card_violations.load(std::memory_order_relaxed);
        return out;
    }

    const BiTHNSWIndex& index() const noexcept { return index_; }
    BiTHNSWIndex& index() noexcept { return index_; }
    std::size_t num_records() const noexcept {
        std::shared_lock<PhaseFairSharedMutex> lk(rw_mu_);
        return records_.size();
    }

    #pragma pack(push, 1)
    struct SnapshotManifest {
        uint64_t magic = 0x5049544D414E4631ULL; // "PITMANF1"
        uint64_t generation = 0;
        uint64_t prov_checkpoint_lsn = 0;
        uint64_t idx_bytes = 0;
        uint64_t prov_bytes = 0;
        uint32_t crc32 = 0;
        uint32_t reserved = 0;
    };

    struct ProvWALHeader {
        uint32_t magic = 0x5056574Cu; // "PVWL"
        uint8_t op = 0;               // 1:INSERT_RAW, 2:REG_PROV, 3:INSERT_REFL, 4:EVOLVE, 5:RESTATE, 6:MAT_REPAIR, 10:BEGIN_TX, 11:COMMIT_TX
        uint8_t reserved[3] = {0, 0, 0};
        uint64_t lsn = 0;
        uint64_t id1 = 0;
        uint64_t id2 = 0;
        uint64_t id3 = 0;
        uint32_t tv_s = 0;
        uint32_t tv_e = 0;
        uint32_t tk_s = 0;
        uint32_t tk_e = 0;
        uint32_t n_parents = 0;
        uint32_t payload_bytes = 0;
        uint32_t crc32 = 0;
    };
    #pragma pack(pop)

    // Atomic full-engine checkpoint: writes generation-stamped .idx + .prov with POSIX fsync,
    // commits via atomic CRC-32 .manifest rename + directory fsync, and refreshes canonical aliases.
    PersistenceStats save_snapshot(const std::string& base_path, bool fsync_disk = true) const {
        std::shared_lock<PhaseFairSharedMutex> lk(rw_mu_);
        auto t0 = std::chrono::steady_clock::now();

        uint64_t prev_gen = 0;
        SnapshotManifest prev_mf{};
        if (read_manifest_file(base_path + ".manifest", prev_mf)) {
            prev_gen = prev_mf.generation;
        }
        const uint64_t next_gen = prev_gen + 1;
        const std::string idx_gen = base_path + ".idx.g" + std::to_string(next_gen);
        const std::string prov_gen = base_path + ".prov.g" + std::to_string(next_gen);

        PersistenceStats st = index_.save_snapshot(idx_gen, fsync_disk);

        const std::string tmp_prov = prov_gen + ".tmp." + std::to_string(static_cast<unsigned long long>(::getpid()));
        int pfd = ::open(tmp_prov.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (pfd < 0) {
            ::unlink(idx_gen.c_str());
            throw std::runtime_error("Cannot open provenance snapshot staging path: " + tmp_prov);
        }

        std::vector<uint8_t> pbuf;
        pbuf.reserve(1024 * 1024);
        std::size_t prov_bytes = 0;

        auto abort_prov = [&](const char* msg) {
            ::close(pfd);
            ::unlink(tmp_prov.c_str());
            ::unlink(idx_gen.c_str());
            throw std::runtime_error(msg);
        };
        auto flush_pbuf = [&]() {
            const uint8_t* ptr = pbuf.data();
            std::size_t rem = pbuf.size();
            while (rem > 0) {
                ssize_t w = ::write(pfd, ptr, rem);
                if (w <= 0) abort_prov("Provenance snapshot write failed");
                ptr += static_cast<std::size_t>(w);
                rem -= static_cast<std::size_t>(w);
                prov_bytes += static_cast<std::size_t>(w);
            }
            pbuf.clear();
        };
        auto emit_raw = [&](const void* src, std::size_t len) {
            if (len == 0) return;
            if (pbuf.size() + len > 1024 * 1024) flush_pbuf();
            if (len > 1024 * 1024) {
                const uint8_t* ptr = static_cast<const uint8_t*>(src);
                std::size_t rem = len;
                while (rem > 0) {
                    ssize_t w = ::write(pfd, ptr, rem);
                    if (w <= 0) abort_prov("Provenance snapshot large write failed");
                    ptr += static_cast<std::size_t>(w);
                    rem -= static_cast<std::size_t>(w);
                    prov_bytes += static_cast<std::size_t>(w);
                }
            } else {
                const uint8_t* ptr = static_cast<const uint8_t*>(src);
                pbuf.insert(pbuf.end(), ptr, ptr + len);
            }
        };
        auto write_pod = [&](const auto& v) { emit_raw(&v, sizeof(v)); };

        const uint64_t magic = 0x50495450524F5632ULL; // "PITPROV2"
        const uint64_t snap_pwal_lsn = pwal_lsn_;
        write_pod(magic);
        write_pod(next_auto_id_);
        write_pod(snap_pwal_lsn);
        uint64_t n_rec = records_.size();
        write_pod(n_rec);
        for (const auto& [rid, r] : records_) {
            write_pod(rid);
            uint8_t tier_u8 = static_cast<uint8_t>(r.tier);
            write_pod(tier_u8);
            write_pod(r.tv_s); write_pod(r.tv_e); write_pod(r.tk_s); write_pod(r.tk_e);
            write_pod(r.suspect_after_tk); write_pod(r.repaired_at_tk);
            uint8_t rstate = r.repair_state->load(std::memory_order_relaxed);
            uint32_t rhot = r.read_hotness_tainted->load(std::memory_order_relaxed);
            write_pod(rstate); write_pod(rhot);
            uint32_t np = static_cast<uint32_t>(r.parents.size());
            write_pod(np);
            if (np > 0) emit_raw(r.parents.data(), np * sizeof(uint64_t));
            uint32_t nc = static_cast<uint32_t>(r.children.size());
            write_pod(nc);
            if (nc > 0) emit_raw(r.children.data(), nc * sizeof(uint64_t));
            uint32_t nw = static_cast<uint32_t>(r.evolve_watermarks.size());
            write_pod(nw);
            for (const auto& ew : r.evolve_watermarks) { write_pod(ew.first); write_pod(ew.second); }
            uint32_t nr = static_cast<uint32_t>(r.replacements.size());
            write_pod(nr);
            for (const auto& rp : r.replacements) { write_pod(rp.first); write_pod(rp.second); }
        }
        uint64_t n_succ = succ_map_.size();
        write_pod(n_succ);
        for (const auto& [oid, pr] : succ_map_) { write_pod(oid); write_pod(pr.first); write_pod(pr.second); }
        uint64_t n_ev = evolve_slices_.size();
        write_pod(n_ev);
        for (const auto& [oid, vec_es] : evolve_slices_) {
            write_pod(oid);
            uint32_t m = static_cast<uint32_t>(vec_es.size());
            write_pod(m);
            if (m > 0) emit_raw(vec_es.data(), m * sizeof(EvolveSplitInfo));
        }
        uint64_t n_cret = cold_retracted_tk_.size();
        write_pod(n_cret);
        for (const auto& [oid, rtk] : cold_retracted_tk_) { write_pod(oid); write_pod(rtk); }

        flush_pbuf();
        if (fsync_disk) ::fsync(pfd);
        ::close(pfd);
        if (::rename(tmp_prov.c_str(), prov_gen.c_str()) != 0) {
            ::unlink(tmp_prov.c_str());
            ::unlink(idx_gen.c_str());
            throw std::runtime_error("Atomic rename of provenance snapshot failed: " + prov_gen);
        }

        // Atomic commit point: write CRC-32 manifest and fsync parent directory
        SnapshotManifest mf{};
        mf.generation = next_gen;
        mf.prov_checkpoint_lsn = snap_pwal_lsn;
        mf.idx_bytes = st.bytes_transferred;
        mf.prov_bytes = prov_bytes;
        mf.crc32 = 0;
        mf.crc32 = crc32_ieee(&mf, sizeof(mf), 0u);

        const std::string mf_path = base_path + ".manifest";
        const std::string mf_tmp = mf_path + ".tmp." + std::to_string(static_cast<unsigned long long>(::getpid()));
        int mfd = ::open(mf_tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (mfd < 0) {
            ::unlink(idx_gen.c_str());
            ::unlink(prov_gen.c_str());
            throw std::runtime_error("Cannot open snapshot manifest staging path: " + mf_tmp);
        }
        if (::write(mfd, &mf, sizeof(mf)) != static_cast<ssize_t>(sizeof(mf))) {
            ::close(mfd);
            ::unlink(mf_tmp.c_str());
            ::unlink(idx_gen.c_str());
            ::unlink(prov_gen.c_str());
            throw std::runtime_error("Snapshot manifest write failed: " + mf_tmp);
        }
        if (fsync_disk) ::fsync(mfd);
        ::close(mfd);
        if (::rename(mf_tmp.c_str(), mf_path.c_str()) != 0) {
            ::unlink(mf_tmp.c_str());
            ::unlink(idx_gen.c_str());
            ::unlink(prov_gen.c_str());
            throw std::runtime_error("Atomic rename of snapshot manifest failed: " + mf_path);
        }

        // Refresh canonical .idx and .prov hardlink aliases and retire superseded generation files
        publish_alias_link(idx_gen, base_path + ".idx");
        publish_alias_link(prov_gen, base_path + ".prov");
        if (prev_gen > 0 && prev_gen != next_gen) {
            ::unlink((base_path + ".idx.g" + std::to_string(prev_gen)).c_str());
            ::unlink((base_path + ".prov.g" + std::to_string(prev_gen)).c_str());
        }
        if (fsync_disk) fsync_parent_dir(mf_path);

        prov_checkpoint_lsn_ = snap_pwal_lsn;
        auto t1 = std::chrono::steady_clock::now();
        st.bytes_transferred += prov_bytes + sizeof(mf);
        st.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        st.throughput_mb_s = (st.elapsed_ms > 0.0)
            ? (static_cast<double>(st.bytes_transferred) / (1024.0 * 1024.0)) / (st.elapsed_ms / 1000.0)
            : 0.0;
        return st;
    }

    PersistenceStats load_snapshot(const std::string& base_path, bool populate_pages = true) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        auto t0 = std::chrono::steady_clock::now();

        std::string idx_path = base_path + ".idx";
        std::string prov_path = base_path + ".prov";
        SnapshotManifest mf{};
        const std::string mf_path = base_path + ".manifest";
        struct stat msb{};
        if (::stat(mf_path.c_str(), &msb) == 0) {
            if (!read_manifest_file(mf_path, mf)) {
                throw std::runtime_error("Corrupt snapshot manifest (CRC-32 mismatch): " + mf_path);
            }
            const std::string g_idx = base_path + ".idx.g" + std::to_string(mf.generation);
            const std::string g_prov = base_path + ".prov.g" + std::to_string(mf.generation);
            struct stat sb1{}, sb2{};
            if (::stat(g_idx.c_str(), &sb1) == 0 && ::stat(g_prov.c_str(), &sb2) == 0) {
                idx_path = g_idx;
                prov_path = g_prov;
            }
        }

        PersistenceStats st = index_.load_snapshot(idx_path, populate_pages);
        std::ifstream ifs(prov_path, std::ios::binary);
        if (!ifs) throw std::runtime_error("Cannot open provenance snapshot: " + prov_path);
        auto read_pod = [&](auto& v) {
            ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
            if (!ifs) throw std::runtime_error("Truncated provenance snapshot: " + prov_path);
        };
        uint64_t magic = 0;
        read_pod(magic);
        if (magic != 0x50495450524F5632ULL && magic != 0x50495450524F5631ULL) {
            throw std::runtime_error("Provenance snapshot magic mismatch");
        }
        read_pod(next_auto_id_);
        uint64_t loaded_lsn = 0;
        if (magic == 0x50495450524F5632ULL) {
            read_pod(loaded_lsn);
        }

        records_.clear();
        succ_map_.clear();
        evolve_slices_.clear();
        cold_retracted_tk_.clear();

        uint64_t n_rec = 0;
        read_pod(n_rec);
        records_.reserve(static_cast<std::size_t>(n_rec));
        for (uint64_t i = 0; i < n_rec; ++i) {
            ProvenanceRecord r;
            uint64_t rid = 0;
            uint8_t tier_u8 = 0, rstate = 0;
            uint32_t rhot = 0;
            read_pod(rid);
            read_pod(tier_u8);
            r.id = rid;
            r.tier = static_cast<MemoryTier>(tier_u8);
            read_pod(r.tv_s); read_pod(r.tv_e); read_pod(r.tk_s); read_pod(r.tk_e);
            read_pod(r.suspect_after_tk); read_pod(r.repaired_at_tk);
            read_pod(rstate); read_pod(rhot);
            r.repair_state->store(rstate, std::memory_order_relaxed);
            r.read_hotness_tainted->store(rhot, std::memory_order_relaxed);
            uint32_t np = 0, nc = 0, nw = 0, nr = 0;
            read_pod(np);
            r.parents.resize(np);
            if (np > 0) ifs.read(reinterpret_cast<char*>(r.parents.data()), np * sizeof(uint64_t));
            read_pod(nc);
            r.children.resize(nc);
            if (nc > 0) ifs.read(reinterpret_cast<char*>(r.children.data()), nc * sizeof(uint64_t));
            read_pod(nw);
            r.evolve_watermarks.resize(nw);
            for (uint32_t j = 0; j < nw; ++j) { read_pod(r.evolve_watermarks[j].first); read_pod(r.evolve_watermarks[j].second); }
            read_pod(nr);
            r.replacements.resize(nr);
            for (uint32_t j = 0; j < nr; ++j) { read_pod(r.replacements[j].first); read_pod(r.replacements[j].second); }
            records_[rid] = std::move(r);
        }
        uint64_t n_succ = 0;
        read_pod(n_succ);
        for (uint64_t i = 0; i < n_succ; ++i) {
            uint64_t oid = 0, nid = 0; uint32_t rtk = 0;
            read_pod(oid); read_pod(nid); read_pod(rtk);
            succ_map_[oid] = {nid, rtk};
        }
        uint64_t n_ev = 0;
        read_pod(n_ev);
        for (uint64_t i = 0; i < n_ev; ++i) {
            uint64_t oid = 0; uint32_t m = 0;
            read_pod(oid); read_pod(m);
            std::vector<EvolveSplitInfo> vec_es(m);
            if (m > 0) ifs.read(reinterpret_cast<char*>(vec_es.data()), m * sizeof(EvolveSplitInfo));
            evolve_slices_[oid] = std::move(vec_es);
        }
        if (magic == 0x50495450524F5632ULL) {
            uint64_t n_cret = 0;
            read_pod(n_cret);
            for (uint64_t i = 0; i < n_cret; ++i) {
                uint64_t oid = 0; uint32_t rtk = 0;
                read_pod(oid); read_pod(rtk);
                cold_retracted_tk_[oid] = rtk;
            }
        }
        if (!ifs) throw std::runtime_error("Truncated provenance snapshot body: " + prov_path);

        prov_checkpoint_lsn_ = (mf.prov_checkpoint_lsn > loaded_lsn) ? mf.prov_checkpoint_lsn : loaded_lsn;
        if (prov_checkpoint_lsn_ > pwal_lsn_) pwal_lsn_ = prov_checkpoint_lsn_;

        struct stat psb{};
        std::size_t prov_sz = (::stat(prov_path.c_str(), &psb) == 0 && psb.st_size > 0)
            ? static_cast<std::size_t>(psb.st_size) : 0u;
        auto t1 = std::chrono::steady_clock::now();
        st.bytes_transferred += prov_sz;
        st.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        st.throughput_mb_s = (st.elapsed_ms > 0.0)
            ? (static_cast<double>(st.bytes_transferred) / (1024.0 * 1024.0)) / (st.elapsed_ms / 1000.0)
            : 0.0;
        return st;
    }

    // Coordinated full-engine append-only CRC-32 WAL (logs both BiT-HNSW vector mutations & provenance DAG ops)
    void open_wal(const std::string& wal_path, bool sync_each_commit = false) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        close_wal_unlocked();
        int rfd = ::open(wal_path.c_str(), O_RDONLY);
        if (rfd >= 0) {
            struct stat sb{};
            if (::fstat(rfd, &sb) == 0 && sb.st_size > 0) {
                std::size_t fsz = static_cast<std::size_t>(sb.st_size);
                std::vector<uint8_t> buf(fsz);
                std::size_t off = 0;
                while (off < fsz) {
                    ssize_t r = ::read(rfd, buf.data() + off, fsz - off);
                    if (r <= 0) break;
                    off += static_cast<std::size_t>(r);
                }
                std::size_t cur = 0;
                while (cur + sizeof(ProvWALHeader) <= off) {
                    ProvWALHeader hdr{};
                    std::memcpy(&hdr, buf.data() + cur, sizeof(hdr));
                    if (hdr.magic != 0x5056574Cu || cur + sizeof(ProvWALHeader) + hdr.payload_bytes > off) break;
                    if (hdr.lsn > pwal_lsn_) pwal_lsn_ = hdr.lsn;
                    cur += sizeof(ProvWALHeader) + hdr.payload_bytes;
                }
            }
            ::close(rfd);
        }
        pwal_fd_ = ::open(wal_path.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
        if (pwal_fd_ < 0) throw std::runtime_error("Failed to open engine WAL file: " + wal_path);
        pwal_sync_each_ = sync_each_commit;
        pwal_tx_depth_ = 0;
    }

    void sync_wal() {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        if (pwal_fd_ >= 0) ::fdatasync(pwal_fd_);
        index_.sync_wal();
    }

    void close_wal() {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        close_wal_unlocked();
    }

    PersistenceStats replay_wal(const std::string& wal_path) {
        std::unique_lock<PhaseFairSharedMutex> lk(rw_mu_);
        auto t0 = std::chrono::steady_clock::now();
        int fd = ::open(wal_path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Failed to open engine WAL for replay: " + wal_path);
        struct stat sb{};
        ::fstat(fd, &sb);
        std::size_t file_sz = static_cast<std::size_t>(std::max<off_t>(0, sb.st_size));
        PersistenceStats st;
        if (file_sz == 0) {
            ::close(fd);
            return st;
        }
        std::vector<uint8_t> buf(file_sz);
        std::size_t off = 0;
        while (off < file_sz) {
            ssize_t r = ::read(fd, buf.data() + off, file_sz - off);
            if (r <= 0) break;
            off += static_cast<std::size_t>(r);
        }
        ::close(fd);

        std::size_t cur = 0;
        std::size_t last_committed_end = 0;
        uint64_t replayed = 0;
        bool in_tx = false;
        std::vector<std::pair<ProvWALHeader, const uint8_t*>> tx_buf;

        while (cur + sizeof(ProvWALHeader) <= off) {
            ProvWALHeader hdr{};
            std::memcpy(&hdr, buf.data() + cur, sizeof(hdr));
            if (hdr.magic != 0x5056574Cu) break;
            if (cur + sizeof(ProvWALHeader) + hdr.payload_bytes > off) break;

            uint32_t saved_crc = hdr.crc32;
            hdr.crc32 = 0;
            uint32_t calc_crc = crc32_ieee(&hdr, sizeof(hdr), 0u);
            const uint8_t* payload = buf.data() + cur + sizeof(hdr);
            if (hdr.payload_bytes > 0) calc_crc = crc32_ieee(payload, hdr.payload_bytes, calc_crc);
            if (calc_crc != saved_crc) break;

            std::size_t next_cur = cur + sizeof(ProvWALHeader) + hdr.payload_bytes;
            if (hdr.lsn > pwal_lsn_) pwal_lsn_ = hdr.lsn;

            if (hdr.op == 10) {
                in_tx = true;
                tx_buf.clear();
            } else if (hdr.op == 11) {
                if (in_tx) {
                    for (const auto& [thdr, tpay] : tx_buf) {
                        if (apply_replayed_prov_op(thdr, tpay)) replayed++;
                    }
                    tx_buf.clear();
                    in_tx = false;
                }
                last_committed_end = next_cur;
            } else {
                if (in_tx) {
                    tx_buf.emplace_back(hdr, payload);
                } else {
                    if (apply_replayed_prov_op(hdr, payload)) replayed++;
                    last_committed_end = next_cur;
                }
            }
            cur = next_cur;
        }

        if (last_committed_end < file_sz) {
            int trc = ::truncate(wal_path.c_str(), static_cast<off_t>(last_committed_end));
            (void)trc;
        }

        auto t1 = std::chrono::steady_clock::now();
        st.bytes_transferred = last_committed_end;
        st.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        st.throughput_mb_s = (st.elapsed_ms > 0.0)
            ? (static_cast<double>(last_committed_end) / (1024.0 * 1024.0)) / (st.elapsed_ms / 1000.0)
            : 0.0;
        st.records_processed = replayed;
        return st;
    }

private:
    mutable PhaseFairSharedMutex rw_mu_;
    BiTHNSWIndex index_;
    std::size_t dim_;
    uint32_t hotness_thresh_;
    uint64_t next_auto_id_ = HIST_SLICE_BASE;
    std::unordered_map<uint64_t, ProvenanceRecord> records_;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint32_t>> succ_map_; // old_id -> (new_id, restate_tk)
    std::unordered_map<uint64_t, std::vector<EvolveSplitInfo>> evolve_slices_; // old_id -> list of EvolveSplitInfo
    std::unordered_map<uint64_t, uint32_t> cold_retracted_tk_; // old_id -> restate/evolve tk_now for cold store overlay
    const RAPCompactStore* cold_store_ = nullptr;

    int pwal_fd_ = -1;
    bool pwal_sync_each_ = false;
    uint32_t pwal_tx_depth_ = 0;
    uint64_t pwal_lsn_ = 0;
    mutable uint64_t prov_checkpoint_lsn_ = 0;

    static uint32_t crc32_ieee(const void* data, std::size_t len, uint32_t crc_init = 0u) noexcept {
        uint32_t crc = ~crc_init;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (std::size_t i = 0; i < len; ++i) {
            crc ^= p[i];
            for (int b = 0; b < 8; ++b) {
                crc = (crc >> 1) ^ (0xEDB88320u & (-(crc & 1u)));
            }
        }
        return ~crc;
    }

    static bool read_manifest_file(const std::string& path, SnapshotManifest& out_mf) noexcept {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        SnapshotManifest mf{};
        ssize_t r = ::read(fd, &mf, sizeof(mf));
        ::close(fd);
        if (r != static_cast<ssize_t>(sizeof(mf)) || mf.magic != 0x5049544D414E4631ULL) return false;
        uint32_t saved_crc = mf.crc32;
        mf.crc32 = 0;
        if (crc32_ieee(&mf, sizeof(mf), 0u) != saved_crc) return false;
        mf.crc32 = saved_crc;
        out_mf = mf;
        return true;
    }

    static void publish_alias_link(const std::string& src_path, const std::string& dst_path) noexcept {
        std::string tmp_lnk = dst_path + ".lnk." + std::to_string(static_cast<unsigned long long>(::getpid()));
        ::unlink(tmp_lnk.c_str());
        if (::link(src_path.c_str(), tmp_lnk.c_str()) == 0) {
            if (::rename(tmp_lnk.c_str(), dst_path.c_str()) != 0) {
                ::unlink(tmp_lnk.c_str());
            }
        }
    }

    static void fsync_parent_dir(const std::string& file_path) noexcept {
        auto slash = file_path.find_last_of('/');
        std::string dir_path = (slash == std::string::npos) ? "." : (slash == 0 ? "/" : file_path.substr(0, slash));
        int dfd = ::open(dir_path.c_str(), O_RDONLY);
        if (dfd >= 0) {
            ::fsync(dfd);
            ::close(dfd);
        }
    }

    void close_wal_unlocked() noexcept {
        if (pwal_fd_ >= 0) {
            ::fdatasync(pwal_fd_);
            ::close(pwal_fd_);
            pwal_fd_ = -1;
        }
        pwal_tx_depth_ = 0;
    }

    void begin_engine_wal_tx() {
        if (pwal_tx_depth_++ == 0) {
            if (pwal_fd_ >= 0) {
                write_pwal_frame(10, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr, 0, false);
            } else {
                index_.begin_wal_tx();
            }
        }
    }

    void commit_engine_wal_tx() {
        if (pwal_tx_depth_ == 0) return;
        if (--pwal_tx_depth_ == 0) {
            if (pwal_fd_ >= 0) {
                write_pwal_frame(11, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr, 0, pwal_sync_each_);
            } else {
                index_.commit_wal_tx();
            }
        }
    }

    void append_prov_wal_op(
        uint8_t op,
        uint64_t id1, uint64_t id2, uint64_t id3,
        uint32_t tv_s, uint32_t tv_e, uint32_t tk_s, uint32_t tk_e,
        const float* vec_opt,
        const uint64_t* parents_opt,
        std::size_t n_parents) {
        if (pwal_fd_ < 0) return;
        write_pwal_frame(op, id1, id2, id3, tv_s, tv_e, tk_s, tk_e, vec_opt, parents_opt, n_parents, false);
    }

    void write_pwal_frame(
        uint8_t op,
        uint64_t id1, uint64_t id2, uint64_t id3,
        uint32_t tv_s, uint32_t tv_e, uint32_t tk_s, uint32_t tk_e,
        const float* vec_opt,
        const uint64_t* parents_opt,
        std::size_t n_parents,
        bool do_sync) {
        if (pwal_fd_ < 0) return;
        const std::size_t vec_bytes = (vec_opt != nullptr) ? (dim_ * sizeof(float)) : 0u;
        const std::size_t par_bytes = (parents_opt != nullptr && n_parents > 0) ? (n_parents * sizeof(uint64_t)) : 0u;

        ProvWALHeader hdr{};
        hdr.op = op;
        hdr.lsn = ++pwal_lsn_;
        hdr.id1 = id1;
        hdr.id2 = id2;
        hdr.id3 = id3;
        hdr.tv_s = tv_s;
        hdr.tv_e = tv_e;
        hdr.tk_s = tk_s;
        hdr.tk_e = tk_e;
        hdr.n_parents = static_cast<uint32_t>(n_parents);
        hdr.payload_bytes = static_cast<uint32_t>(vec_bytes + par_bytes);
        hdr.crc32 = 0;

        uint32_t crc = crc32_ieee(&hdr, sizeof(hdr), 0u);
        if (vec_bytes > 0) crc = crc32_ieee(vec_opt, vec_bytes, crc);
        if (par_bytes > 0) crc = crc32_ieee(parents_opt, par_bytes, crc);
        hdr.crc32 = crc;

        std::vector<uint8_t> frame(sizeof(hdr) + vec_bytes + par_bytes);
        std::memcpy(frame.data(), &hdr, sizeof(hdr));
        if (vec_bytes > 0) std::memcpy(frame.data() + sizeof(hdr), vec_opt, vec_bytes);
        if (par_bytes > 0) std::memcpy(frame.data() + sizeof(hdr) + vec_bytes, parents_opt, par_bytes);

        const uint8_t* ptr = frame.data();
        std::size_t rem = frame.size();
        while (rem > 0) {
            ssize_t w = ::write(pwal_fd_, ptr, rem);
            if (w <= 0) throw std::runtime_error("Engine WAL write failed");
            ptr += static_cast<std::size_t>(w);
            rem -= static_cast<std::size_t>(w);
        }
        if (do_sync) ::fdatasync(pwal_fd_);
    }

    bool apply_replayed_prov_op(const ProvWALHeader& hdr, const uint8_t* payload) {
        if (hdr.lsn <= prov_checkpoint_lsn_) return false;
        const std::size_t vec_bytes = dim_ * sizeof(float);
        const std::size_t par_bytes = static_cast<std::size_t>(hdr.n_parents) * sizeof(uint64_t);

        if (hdr.op == 1 && hdr.payload_bytes == vec_bytes) {
            if (records_.find(hdr.id1) != records_.end()) return false;
            const float* vec = reinterpret_cast<const float*>(payload);
            index_.insert_single(hdr.id1, vec, hdr.tv_s, hdr.tv_e, hdr.tk_s, hdr.tk_e);
            ProvenanceRecord rec;
            rec.id = hdr.id1;
            rec.tier = MemoryTier::RAW;
            rec.tv_s = hdr.tv_s; rec.tv_e = hdr.tv_e;
            rec.tk_s = hdr.tk_s; rec.tk_e = hdr.tk_e;
            records_[hdr.id1] = std::move(rec);
            if (hdr.id1 >= HIST_SLICE_BASE && hdr.id1 >= next_auto_id_) next_auto_id_ = hdr.id1 + 1;
            return true;
        } else if (hdr.op == 2 && hdr.payload_bytes == par_bytes) {
            auto it = records_.find(hdr.id1);
            if (it == records_.end()) return false;
            std::vector<uint64_t> parents(hdr.n_parents);
            if (par_bytes > 0) std::memcpy(parents.data(), payload, par_bytes);
            it->second.parents = std::move(parents);
            bind_parents_and_inherit_taint(it->second);
            return true;
        } else if (hdr.op == 3 && hdr.payload_bytes == vec_bytes + par_bytes) {
            if (records_.find(hdr.id1) != records_.end()) return false;
            const float* vec = reinterpret_cast<const float*>(payload);
            std::vector<uint64_t> parents(hdr.n_parents);
            if (par_bytes > 0) std::memcpy(parents.data(), payload + vec_bytes, par_bytes);
            insert_reflection_unlocked(hdr.id1, vec, hdr.tv_s, hdr.tv_e, hdr.tk_s, hdr.tk_e, parents);
            return true;
        } else if (hdr.op == 4 && hdr.payload_bytes == vec_bytes) {
            auto it = records_.find(hdr.id1);
            if (it == records_.end() || records_.find(hdr.id2) != records_.end()) return false;
            const uint64_t old_id = hdr.id1, new_id = hdr.id2, hist_slice_id = hdr.id3;
            const uint32_t tv_chg = hdr.tv_s, old_tve = hdr.tv_e, tk_now = hdr.tk_s;
            const uint32_t old_tvs = it->second.tv_s;
            const MemoryTier old_tier = it->second.tier;
            std::vector<uint64_t> old_parents = it->second.parents;
            std::vector<uint64_t> old_children = it->second.children;

            index_.update_tx_end(old_id, tk_now);
            it->second.tk_e = tk_now;
            cold_retracted_tk_[old_id] = tk_now;

            const float* old_vp = index_.get_vector_ptr(old_id);
            std::vector<float> old_v_copy(dim_, 0.0f);
            if (old_vp != nullptr) std::memcpy(old_v_copy.data(), old_vp, vec_bytes);
            index_.insert_single(hist_slice_id, old_v_copy.data(), old_tvs, tv_chg, tk_now, INF_TICK);
            ProvenanceRecord hrec;
            hrec.id = hist_slice_id; hrec.tier = old_tier;
            hrec.tv_s = old_tvs; hrec.tv_e = tv_chg; hrec.tk_s = tk_now; hrec.tk_e = INF_TICK;
            hrec.parents = old_parents;
            for (uint64_t cid : old_children) {
                auto cit = records_.find(cid);
                if (cit != records_.end() && cit->second.tv_s < tv_chg && old_tvs < cit->second.tv_e) {
                    add_unique_child(hrec.children, cid);
                }
            }
            records_[hist_slice_id] = std::move(hrec);
            bind_parents_and_inherit_taint(records_[hist_slice_id]);
            evolve_slices_[old_id].push_back(EvolveSplitInfo{hist_slice_id, new_id, tv_chg, tk_now});

            const float* new_vec = reinterpret_cast<const float*>(payload);
            index_.insert_single(new_id, new_vec, tv_chg, old_tve, tk_now, INF_TICK);
            ProvenanceRecord nrec;
            nrec.id = new_id; nrec.tier = old_tier;
            nrec.tv_s = tv_chg; nrec.tv_e = old_tve; nrec.tk_s = tk_now; nrec.tk_e = INF_TICK;
            nrec.parents = old_parents;
            for (uint64_t cid : old_children) {
                auto cit = records_.find(cid);
                if (cit != records_.end() && cit->second.tv_s < old_tve && tv_chg < cit->second.tv_e) {
                    add_unique_child(nrec.children, cid);
                }
            }
            records_[new_id] = std::move(nrec);
            bind_parents_and_inherit_taint(records_[new_id]);
            if (hist_slice_id >= HIST_SLICE_BASE && hist_slice_id >= next_auto_id_) next_auto_id_ = hist_slice_id + 1;
            if (new_id >= HIST_SLICE_BASE && new_id >= next_auto_id_) next_auto_id_ = new_id + 1;

            std::queue<uint64_t> e_bfs;
            std::unordered_set<uint64_t> e_seen;
            for (uint64_t cid : old_children) {
                if (e_seen.insert(cid).second) e_bfs.push(cid);
            }
            while (!e_bfs.empty()) {
                uint64_t cid = e_bfs.front(); e_bfs.pop();
                auto cit = records_.find(cid);
                if (cit == records_.end()) continue;
                auto& crec = cit->second;
                if (crec.tier == MemoryTier::REFLECT && crec.tk_e > tk_now &&
                    crec.tv_e > tv_chg && crec.tv_s < old_tve) {
                    add_unique_watermark(crec.evolve_watermarks, tv_chg, tk_now);
                    add_unique_child(records_[new_id].children, cid);
                }
                for (uint64_t gc : crec.children) {
                    if (e_seen.insert(gc).second) e_bfs.push(gc);
                }
            }
            return true;
        } else if (hdr.op == 5 && hdr.payload_bytes == vec_bytes) {
            if (records_.find(hdr.id2) != records_.end()) return false;
            const uint64_t old_id = hdr.id1, new_id = hdr.id2;
            const bool in_hot = (hdr.id3 != 0);
            const uint32_t orig_tvs = hdr.tv_s, orig_tve = hdr.tv_e, tk_now = hdr.tk_s;
            MemoryTier orig_tier = MemoryTier::RAW;
            std::vector<uint64_t> inherited_children;
            auto it = records_.find(old_id);
            if (in_hot && it != records_.end()) {
                orig_tier = it->second.tier;
                inherited_children = it->second.children;
                index_.update_tx_end(old_id, tk_now);
                it->second.tk_e = tk_now;
            }
            cold_retracted_tk_[old_id] = tk_now;
            succ_map_[old_id] = {new_id, tk_now};

            const float* corr_vec = reinterpret_cast<const float*>(payload);
            index_.insert_single(new_id, corr_vec, orig_tvs, orig_tve, tk_now, INF_TICK);
            ProvenanceRecord nrec;
            nrec.id = new_id; nrec.tier = orig_tier;
            nrec.tv_s = orig_tvs; nrec.tv_e = orig_tve; nrec.tk_s = tk_now; nrec.tk_e = INF_TICK;
            nrec.children = inherited_children;
            records_[new_id] = std::move(nrec);
            if (new_id >= HIST_SLICE_BASE && new_id >= next_auto_id_) next_auto_id_ = new_id + 1;

            std::queue<uint64_t> bfs;
            std::unordered_set<uint64_t> seen;
            for (uint64_t cid : inherited_children) {
                if (seen.insert(cid).second) bfs.push(cid);
            }
            while (!bfs.empty()) {
                uint64_t cid = bfs.front(); bfs.pop();
                auto cit = records_.find(cid);
                if (cit == records_.end()) continue;
                auto& crec = cit->second;
                if (crec.tk_e > tk_now && crec.tv_s < orig_tve && orig_tvs < crec.tv_e) {
                    if (tk_now < crec.suspect_after_tk) crec.suspect_after_tk = tk_now;
                    crec.replacements.emplace_back(old_id, new_id);
                    add_unique_child(records_[new_id].children, cid);
                }
                for (uint64_t gc : crec.children) {
                    if (seen.insert(gc).second) bfs.push(gc);
                }
            }
            return true;
        } else if (hdr.op == 6 && hdr.payload_bytes == vec_bytes + par_bytes) {
            auto it = records_.find(hdr.id1);
            if (it == records_.end() || records_.find(hdr.id2) != records_.end()) return false;
            const uint64_t dirty_id = hdr.id1, new_id = hdr.id2;
            const uint32_t tvs = hdr.tv_s, tve = hdr.tv_e, mat_tk = hdr.tk_s;
            std::vector<uint64_t> inherited_children = it->second.children;
            const float* vec = reinterpret_cast<const float*>(payload);
            std::vector<uint64_t> clean_parents(hdr.n_parents);
            if (par_bytes > 0) std::memcpy(clean_parents.data(), payload + vec_bytes, par_bytes);

            index_.update_tx_end(dirty_id, mat_tk);
            it->second.tk_e = mat_tk;
            it->second.repaired_at_tk = mat_tk;
            it->second.repair_state->store(2, std::memory_order_release);
            succ_map_[dirty_id] = {new_id, mat_tk};

            insert_reflection_unlocked(new_id, vec, tvs, tve, mat_tk, INF_TICK, clean_parents);
            for (uint64_t child_id : inherited_children) {
                add_unique_child(records_[new_id].children, child_id);
                auto cit = records_.find(child_id);
                if (cit != records_.end()) {
                    cit->second.replacements.emplace_back(dirty_id, new_id);
                }
            }
            return true;
        }
        return false;
    }

    static void add_unique_child(std::vector<uint64_t>& vec, uint64_t cid) {
        if (std::find(vec.begin(), vec.end(), cid) == vec.end()) {
            vec.push_back(cid);
        }
    }

    static void add_unique_watermark(
        std::vector<std::pair<uint32_t, uint32_t>>& vec,
        uint32_t tv_chg,
        uint32_t tk_ev) {
        auto p = std::make_pair(tv_chg, tk_ev);
        if (std::find(vec.begin(), vec.end(), p) == vec.end()) {
            vec.push_back(p);
        }
    }

    // Recursively expand multi-hop EVOLVE splits and RESTATE replacement chains up to tk_q
    void expand_lineage_versions(uint64_t root_id, uint32_t tk_q, std::vector<uint64_t>& out_versions) const {
        std::queue<uint64_t> q;
        std::unordered_set<uint64_t> vis;
        if (vis.insert(root_id).second) q.push(root_id);

        while (!q.empty()) {
            uint64_t cur = q.front();
            q.pop();
            out_versions.push_back(cur);

            auto sit = succ_map_.find(cur);
            if (sit != succ_map_.end()) {
                auto [nxt_id, restate_tk] = sit->second;
                if (restate_tk <= tk_q && vis.insert(nxt_id).second) {
                    q.push(nxt_id);
                }
            }
            auto eit = evolve_slices_.find(cur);
            if (eit != evolve_slices_.end()) {
                for (const auto& es : eit->second) {
                    if (es.tk_evolve <= tk_q) {
                        if (vis.insert(es.hist_id).second) q.push(es.hist_id);
                        if (vis.insert(es.new_id).second) q.push(es.new_id);
                    }
                }
            }
        }
    }

    // Bind reverse parent->child edges and inherit taint across the full transitive ancestor closure P*(rec)!
    void bind_parents_and_inherit_taint(ProvenanceRecord& rec) {
        const uint64_t rid = rec.id;
        for (uint64_t pid : rec.parents) {
            auto pit = records_.find(pid);
            if (pit != records_.end()) {
                add_unique_child(pit->second.children, rid);
            }
        }
        if (rec.tier != MemoryTier::REFLECT) return;

        std::queue<uint64_t> anc_q;
        std::unordered_set<uint64_t> anc_seen;
        for (uint64_t pid : rec.parents) {
            if (anc_seen.insert(pid).second) anc_q.push(pid);
        }

        while (!anc_q.empty()) {
            uint64_t aid = anc_q.front();
            anc_q.pop();
            auto ait = records_.find(aid);
            if (ait == records_.end()) continue;
            const auto& arec = ait->second;

            // Always traverse upstream parents even across disjoint-scope intermediate summaries!
            for (uint64_t gpid : arec.parents) {
                if (anc_seen.insert(gpid).second) anc_q.push(gpid);
            }

            const bool overlaps_anc = (rec.tv_s < arec.tv_e) && (arec.tv_s < rec.tv_e);
            if (overlaps_anc) {
                if (arec.tier == MemoryTier::REFLECT) {
                    if (arec.suspect_after_tk < rec.suspect_after_tk) {
                        rec.suspect_after_tk = arec.suspect_after_tk;
                    }
                    for (const auto& ew : arec.evolve_watermarks) {
                        if (ew.first < rec.tv_e) {
                            add_unique_watermark(rec.evolve_watermarks, ew.first, ew.second);
                        }
                    }
                }
                auto sit = succ_map_.find(aid);
                if (sit != succ_map_.end()) {
                    uint32_t first_restate_tk = sit->second.second;
                    if (first_restate_tk < rec.suspect_after_tk) {
                        rec.suspect_after_tk = first_restate_tk;
                    }
                }
            }

            auto eit = evolve_slices_.find(aid);
            if (eit != evolve_slices_.end()) {
                for (const auto& es : eit->second) {
                    uint32_t eff_end = arec.tv_e;
                    auto nit = records_.find(es.new_id);
                    if (nit != records_.end() && nit->second.tv_e > eff_end) {
                        eff_end = nit->second.tv_e;
                    }
                    if (rec.tv_e > es.tv_chg && rec.tv_s < eff_end) {
                        add_unique_watermark(rec.evolve_watermarks, es.tv_chg, es.tk_evolve);
                    }
                }
            }

            // Subscribe rid to all active lineage descendants of aid whose valid window overlaps rec
            if (succ_map_.count(aid) || evolve_slices_.count(aid)) {
                std::vector<uint64_t> lin_vers;
                expand_lineage_versions(aid, INF_TICK - 1, lin_vers);
                for (uint64_t vid : lin_vers) {
                    auto vit = records_.find(vid);
                    if (vit != records_.end() && vit->second.tk_e == INF_TICK &&
                        rec.tv_s < vit->second.tv_e && vit->second.tv_s < rec.tv_e) {
                        add_unique_child(vit->second.children, rid);
                    }
                }
            }
        }
    }

    void insert_reflection_unlocked(
        uint64_t reflection_id,
        const float* vec,
        uint32_t tv_s, uint32_t tv_e,
        uint32_t tk_s, uint32_t tk_e,
        const std::vector<uint64_t>& parent_ids) {
        uint32_t safe_tks = tk_s;
        for (uint64_t pid : parent_ids) {
            auto pit = records_.find(pid);
            if (pit != records_.end() && pit->second.tk_s > safe_tks) {
                safe_tks = pit->second.tk_s;
            }
        }
        index_.insert_single(reflection_id, vec, tv_s, tv_e, safe_tks, tk_e);
        ProvenanceRecord rec;
        rec.id = reflection_id;
        rec.tier = MemoryTier::REFLECT;
        rec.tv_s = tv_s;
        rec.tv_e = tv_e;
        rec.tk_s = safe_tks;
        rec.tk_e = tk_e;
        rec.parents = parent_ids;
        records_[reflection_id] = std::move(rec);
        bind_parents_and_inherit_taint(records_[reflection_id]);
        if (reflection_id >= HIST_SLICE_BASE && reflection_id >= next_auto_id_) next_auto_id_ = reflection_id + 1;
    }

    void collect_clean_parents_for_repair(
        uint64_t refl_id,
        uint32_t req_tvs,
        uint32_t req_tve,
        uint32_t query_tk,
        std::unordered_set<uint64_t>& visited_refl,
        std::vector<uint64_t>& out_parents,
        uint32_t& max_parent_tks) const {
        if (!visited_refl.insert(refl_id).second) return;
        auto it = records_.find(refl_id);
        if (it == records_.end()) return;
        const auto& rec = it->second;

        for (uint64_t pid : rec.parents) {
            std::vector<uint64_t> cand_versions;
            expand_lineage_versions(pid, query_tk, cand_versions);
            for (uint64_t vid : cand_versions) {
                auto rit = records_.find(vid);
                if (rit == records_.end()) continue;
                const auto& rrec = rit->second;
                if (rrec.tk_s > query_tk || rrec.tk_e <= query_tk) continue;
                if (!(rrec.tv_s < req_tve && req_tvs < rrec.tv_e)) continue;

                if (rrec.is_tainted_on_window(req_tvs, req_tve, query_tk)) {
                    // Intermediate reflection is itself tainted over [req_tvs, req_tve) at query_tk -> recursively flatten!
                    collect_clean_parents_for_repair(vid, req_tvs, req_tve, query_tk, visited_refl, out_parents, max_parent_tks);
                } else {
                    if (std::find(out_parents.begin(), out_parents.end(), vid) == out_parents.end()) {
                        out_parents.push_back(vid);
                        if (rrec.tk_s > max_parent_tks) max_parent_tks = rrec.tk_s;
                    }
                }
            }
        }
    }

    const float* lookup_record_vec(uint64_t id) const noexcept {
        return index_.get_vector_ptr(id);
    }

    void unpack_clean_evidence(
        uint64_t refl_id,
        uint32_t tv_q,
        uint32_t tk_q,
        std::unordered_set<uint64_t>& seen_cands,
        std::unordered_set<uint64_t>& unpack_vis,
        std::vector<uint64_t>& out_clean_ids) const {
        if (!unpack_vis.insert(refl_id).second) return;
        auto it = records_.find(refl_id);
        if (it == records_.end()) return;
        const auto& rec = it->second;

        for (uint64_t pid : rec.parents) {
            std::vector<uint64_t> cand_versions;
            expand_lineage_versions(pid, tk_q, cand_versions);
            for (uint64_t target_id : cand_versions) {
                auto tit = records_.find(target_id);
                if (tit == records_.end()) continue;
                const auto& tr = tit->second;
                if (bitemp_visible(tr.tv_s, tr.tv_e, tr.tk_s, tr.tk_e, tv_q, tk_q)) {
                    if (tr.is_tainted_at(tv_q, tk_q)) {
                        unpack_clean_evidence(target_id, tv_q, tk_q, seen_cands, unpack_vis, out_clean_ids);
                    } else if (seen_cands.insert(target_id).second) {
                        out_clean_ids.push_back(target_id);
                    }
                }
            }
        }
    }
};

} // namespace pitmem
