#pragma once

#include "pit_types.hpp"
#include "simd_kernels.hpp"

#include <vector>
#include <array>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pitmem {

// Lightweight spinlock for fine-grained per-node adjacency updates during parallel build
struct SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    inline void lock() noexcept {
        while (flag.test_and_set(std::memory_order_acquire)) {
        #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
        #endif
        }
    }
    inline void unlock() noexcept {
        flag.clear(std::memory_order_release);
    }
};

// Pack 24-bit segment-local node ID (up to 16.77M nodes per MacroSegment) + 4-bit gv + 4-bit gk into uint32_t.
// Global capacity across MacroSegments is unbounded (64-bit external IDs & 64-bit segment:local handles).
inline uint32_t pack_neighbor(uint32_t local_id, uint8_t gv, uint8_t gk) noexcept {
    return (local_id & 0x00FFFFFFu)
        | (static_cast<uint32_t>(gv & 0x0Fu) << 24)
        | (static_cast<uint32_t>(gk & 0x0Fu) << 28);
}
inline uint32_t unpack_id(uint32_t packed) noexcept { return packed & 0x00FFFFFFu; }
inline uint8_t unpack_gv(uint32_t packed) noexcept { return static_cast<uint8_t>((packed >> 24) & 0x0Fu); }
inline uint8_t unpack_gk(uint32_t packed) noexcept { return static_cast<uint8_t>((packed >> 28) & 0x0Fu); }

// IEEE 802.3 CRC-32 checksum (polynomial 0xEDB88320) for WAL record integrity verification
inline uint32_t crc32_ieee(const void* data, std::size_t len, uint32_t prev_crc = 0u) noexcept {
    static const auto table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint32_t crc = ~prev_crc;
    for (std::size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return ~crc;
}

struct BiTHNSWConfig {
    std::size_t dim = 768;
    uint32_t m_pre = 16;          // Causal-Prefix degree cap (never overwritten by future cells)
    uint32_t m_fwd = 12;          // Forward-Bridge degree cap (for cross-epoch upward navigation)
    uint32_t m_upper = 16;        // Upper HNSW layer degree cap
    uint32_t ef_construction = 96;
    uint32_t grid_bins = 16;      // Up to 16x16 2D Birth-Epoch Grid per MacroSegment
    uint64_t scan_alive_thresh = 8000;       // Legacy floor knob (superseded by analytical multi-segment cost router)
    float min_graph_survivor_ratio = 0.055f; // Switch to SIMD SQ8 scan when global prefix survivor density < 5.5%
    bool streaming_mode = true;              // Online zero-look-ahead causal segment partitioning
    uint64_t initial_segment_cap = 131072;   // Geometric segment rollover: 128K -> 256K -> max_segment_cap
    uint64_t max_segment_cap = 425984;       // Bounded sealed MacroSegment cap (416K rows) for balanced tail epochs & NVMe mmap offload
    // Hardware-calibrated analytical router parameters (normalized to one 768-d maddubs evaluation; see calibrate_router_hardware)
    bool auto_calibrate_router = false;
    double router_gamma_0 = 140.0;
    double router_gamma_ts = 0.125;
    double router_gamma_hub = 1050.0;
    double router_gamma_llc = 6.25;
    double router_gamma_dim = 10.8;
    double router_llc_knee = 450000.0;
};

struct PersistenceStats {
    std::size_t bytes_transferred = 0;
    double elapsed_ms = 0.0;
    double throughput_mb_s = 0.0;
    uint64_t records_processed = 0;
};

class BiTHNSWIndex {
public:
    static constexpr uint32_t MAX_BINS = 16;
    static constexpr uint32_t MORT_BINS = 16;
    static constexpr uint32_t MAX_SEG_LOCAL_NODES = 16000000u; // < 2^24 (16,777,215)

    struct EpochCell {
        uint8_t gv = 0;
        uint8_t gk = 0;
        uint32_t min_tv_s = INF_TICK;
        uint32_t max_tv_s = 0;
        uint32_t min_tk_s = INF_TICK;
        uint32_t max_tk_s = 0;
        uint32_t max_tv_e = 0;
        uint32_t max_tk_e = 0;

        // Single canonical cell-contiguous storage shared by BOTH AVX2 Scan and Graph Traversal
        // (eliminates duplicate global SQ8 arrays, saving 784 B/record = 1.5 GiB at N=2M!)
        AlignedBuffer<uint32_t, 64> tv_s;
        AlignedBuffer<uint32_t, 64> tv_e;
        AlignedBuffer<uint32_t, 64> tk_s;
        AlignedBuffer<uint32_t, 64> tk_e;
        std::vector<uint32_t> local_ids; // maps cell slot -> segment-local node_id
        AlignedBuffer<int8_t, 64> sq8_codes;
        std::vector<simd::SQ8VectorMeta> sq8_meta;

        // O(1) Mortality Histogram for exact cell-dead pruning & survivor estimation
        uint32_t tv_end_hist[MORT_BINS] = {0};
        uint32_t tk_end_hist[MORT_BINS] = {0};

        // Stratified intra-cell graph entry hubs
        std::vector<uint32_t> entry_nodes;

        std::size_t size() const noexcept { return local_ids.size(); }
    };

    struct MacroSegment {
        uint32_t seg_id = 0;
        uint64_t capacity = 131072;
        uint64_t epoch_row_cap = 8192;
        bool sealed = false;
        bool bounds_ready = false;

        uint32_t min_tv_s = INF_TICK;
        uint32_t max_tv_s = 0;
        uint32_t min_tk_s = INF_TICK;
        uint32_t max_tk_s = 0;
        uint32_t max_tv_e = 0;
        uint32_t max_tk_e = 0;

        uint32_t mort_min_tick = 0;
        uint32_t mort_max_tick = 1000000;

        // Causal valid-time split bounds (fixed at segment creation from prior prefix; never rewired)
        std::vector<uint32_t> v_bounds;
        // Online transaction-time column progression: active_gk advances as each column hits epoch_row_cap
        uint8_t active_gk = 0;
        uint64_t gk_counts[MAX_BINS] = {0};
        uint64_t gv_counts[MAX_BINS] = {0};
        uint32_t k_max_per_col[MAX_BINS] = {0};
        uint32_t k_min_per_col[MAX_BINS] = {0};

        std::size_t num_nodes = 0;
        std::atomic<std::size_t> linked_count{0};

        std::vector<uint64_t> ext_ids;
        std::vector<uint8_t> node_gv;
        std::vector<uint8_t> node_gk;
        std::vector<uint8_t> node_cid;
        std::vector<uint32_t> node_cell_slot;
        std::vector<uint8_t> node_level;

        // Compact 16-byte L1-friendly per-node timestamp mirror for 1-hop graph predicate checks
        AlignedBuffer<uint32_t, 64> g_tv_s;
        AlignedBuffer<uint32_t, 64> g_tv_e;
        AlignedBuffer<uint32_t, 64> g_tk_s;
        AlignedBuffer<uint32_t, 64> g_tk_e;
        AlignedBuffer<float, 64> fp32_vecs;

        // Optional NVMe mmap cold-FP32 backing (shrinks resident RAM from 4,105 B/rec to 1,033 B/rec!)
        int fp32_mmap_fd = -1;
        float* fp32_mmap_ptr = nullptr;
        std::size_t fp32_mmap_bytes = 0;
        std::string fp32_mmap_path;

        inline const float* fp32_ptr(std::size_t lid, std::size_t dim) const noexcept {
            const float* base = (fp32_mmap_ptr != nullptr) ? fp32_mmap_ptr : fp32_vecs.data;
            return base + lid * dim;
        }

        std::vector<std::vector<uint32_t>> adj_pre;
        std::vector<std::vector<uint32_t>> adj_fwd;
        std::vector<std::vector<std::vector<uint32_t>>> adj_upper;
        std::vector<SpinLock> node_locks;

        std::vector<EpochCell> cells;
        SpinLock ready_lock;
        std::vector<uint32_t> cell_seeds[MAX_BINS * MAX_BINS];
        std::vector<uint32_t> upper_hubs;

        std::vector<uint32_t> single_vis;
        uint32_t single_vis_tok = 1;

        explicit MacroSegment(uint32_t id, uint32_t bins, uint64_t cap)
            : seg_id(id), capacity(std::min<uint64_t>(cap, MAX_SEG_LOCAL_NODES)) {
            epoch_row_cap = std::max<uint64_t>(64ULL, (capacity + bins - 1) / bins);
            cells.resize(bins * bins);
            for (uint32_t gk = 0; gk < bins; ++gk) {
                k_min_per_col[gk] = INF_TICK;
                k_max_per_col[gk] = 0;
                for (uint32_t gv = 0; gv < bins; ++gv) {
                    auto& c = cells[gk * bins + gv];
                    c.gv = static_cast<uint8_t>(gv);
                    c.gk = static_cast<uint8_t>(gk);
                }
            }
        }

        ~MacroSegment() {
            #if defined(__unix__) || defined(__APPLE__)
            if (fp32_mmap_ptr != nullptr && fp32_mmap_bytes > 0) {
                ::munmap(fp32_mmap_ptr, fp32_mmap_bytes);
                fp32_mmap_ptr = nullptr;
                fp32_mmap_bytes = 0;
            }
            if (fp32_mmap_fd >= 0) {
                ::close(fp32_mmap_fd);
                fp32_mmap_fd = -1;
            }
            #endif
        }
    };

    explicit BiTHNSWIndex(const BiTHNSWConfig& cfg)
        : cfg_(cfg), padded_dim_(((cfg.dim + 31) / 32) * 32) {
        if (cfg_.grid_bins < 2 || cfg_.grid_bins > MAX_BINS) {
            throw std::invalid_argument("grid_bins must be in [2, 16]");
        }
        level_mult_ = 1.0 / std::log(static_cast<double>(std::max<uint32_t>(2, cfg_.m_pre)));
        if (cfg_.auto_calibrate_router) {
            calibrate_router_hardware();
        }
    }
    BiTHNSWIndex(const BiTHNSWIndex&) = delete;
    BiTHNSWIndex& operator=(const BiTHNSWIndex&) = delete;

    ~BiTHNSWIndex() {
        close_wal();
    }

    const BiTHNSWConfig& config() const noexcept { return cfg_; }

    // Millisecond startup hardware micro-calibration for Eq. (3) analytical router ratios:
    // Normalizes 8-wide unsigned AVX2 timestamp checks and random LLC/DRAM pointer-chasing hops
    // against sequential 768-d maddubs SQ8 inner-product bandwidth on the host CPU.
    BiTHNSWConfig calibrate_router_hardware(std::size_t l3_cache_bytes = 47185920ULL) {
        constexpr std::size_t kRefDim = 768;
        constexpr std::size_t kSeqSlots = 8192;
        constexpr std::size_t kRandNodes = 16384; // ~16 MiB SQ8+adj working set spanning L3/DRAM

        std::vector<float> qf(kRefDim, 0.035f);
        simd::SQ8Query sq_q = simd::prepare_sq8_query(qf.data(), kRefDim);
        AlignedBuffer<int8_t, 64> seq_codes;
        seq_codes.resize(kSeqSlots * kRefDim);
        std::vector<simd::SQ8VectorMeta> seq_meta(kSeqSlots);
        for (std::size_t i = 0; i < kSeqSlots * kRefDim; ++i) {
            seq_codes.data[i] = static_cast<int8_t>((i * 37u) % 121u - 60);
        }
        for (std::size_t i = 0; i < kSeqSlots; ++i) {
            seq_meta[i] = {0.01f, 1.0f, 0};
        }

        AlignedBuffer<uint32_t, 64> ts_a, ts_b, ts_c, ts_d;
        ts_a.resize(kSeqSlots, 100u); ts_b.resize(kSeqSlots, INF_TICK);
        ts_c.resize(kSeqSlots, 100u); ts_d.resize(kSeqSlots, INF_TICK);

        // 1) Sequential 768-d maddubs SQ8 baseline cost
        volatile float sink_f = 0.0f;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 4; ++rep) {
            float acc = 0.0f;
            for (std::size_t i = 0; i < kSeqSlots; ++i) {
                acc += simd::sq8_approx_l2_sqr(sq_q, seq_codes.data + i * kRefDim, seq_meta[i]);
            }
            sink_f = acc;
        }
        double ns_per_maddubs = std::max(
            1.0,
            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count()
                / static_cast<double>(4 * kSeqSlots));

        // 2) Sequential 8-slot AVX2 timestamp check cost
        volatile uint64_t sink_u = 0;
        auto t1 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 16; ++rep) {
            uint64_t msum = 0;
            for (std::size_t base = 0; base < kSeqSlots; base += 8) {
                msum += simd::filter8_full_bitemp(
                    ts_a.data + base, ts_b.data + base, ts_c.data + base, ts_d.data + base, 500u, 500u);
            }
            sink_u = msum;
        }
        double ns_per_ts8 = std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - t1).count() / static_cast<double>(16 * (kSeqSlots / 8));

        // 3) Random pointer-chasing + SQ8 evaluation across kRandNodes permutation ring
        AlignedBuffer<int8_t, 64> rand_codes;
        rand_codes.resize(kRandNodes * kRefDim);
        std::memcpy(rand_codes.data, seq_codes.data, std::min(kSeqSlots, kRandNodes) * kRefDim);
        std::vector<uint32_t> perm(kRandNodes), next_hop(kRandNodes);
        std::iota(perm.begin(), perm.end(), 0u);
        std::mt19937 rng(42);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (std::size_t i = 0; i < kRandNodes; ++i) {
            next_hop[perm[i]] = perm[(i + 1) % kRandNodes];
        }
        uint32_t cur = perm[0];
        auto t2 = std::chrono::steady_clock::now();
        float racc = 0.0f;
        for (std::size_t step = 0; step < kRandNodes * 2; ++step) {
            cur = next_hop[cur];
            racc += simd::sq8_approx_l2_sqr(sq_q, rand_codes.data + static_cast<std::size_t>(cur) * kRefDim, seq_meta[0]);
        }
        sink_f = racc + static_cast<float>(sink_u & 1u);
        if (sink_f == -1e30f) return cfg_;
        double ns_per_rand_hop = std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - t2).count() / static_cast<double>(kRandNodes * 2);

        double hop_ratio = std::clamp(ns_per_rand_hop / ns_per_maddubs, 1.5, 12.0);
        // Scale per-beam-unit graph expansion (evaluating ~5-6 diversified neighbors per beam step)
        double total_beam_unit = std::clamp(hop_ratio * 5.2, 8.0, 42.0);
        cfg_.router_gamma_ts = std::clamp(ns_per_ts8 / ns_per_maddubs, 0.04, 0.45);
        cfg_.router_gamma_llc = total_beam_unit * 0.37;
        cfg_.router_gamma_dim = total_beam_unit * 0.63;
        cfg_.router_gamma_hub = std::clamp(total_beam_unit * 60.0, 400.0, 2400.0);
        cfg_.router_gamma_0 = std::clamp(total_beam_unit * 8.0, 60.0, 320.0);
        // Core hot index arrays occupy ~1,032 B/rec; DRAM transition saturates around ~10x L3 capacity
        if (l3_cache_bytes > 0) {
            cfg_.router_llc_knee = std::clamp(
                (static_cast<double>(l3_cache_bytes) / 1032.0) * 9.8, 100000.0, 2500000.0);
        }
        return cfg_;
    }

    // Legacy explicit calibration hook (used only when streaming_mode=false for static/oracle ablations)
    void calibrate_bounds(const uint32_t* tv_s, const uint32_t* tk_s, std::size_t n) {
        if (n == 0) return;
        std::size_t sample_n = std::min<std::size_t>(n, 65536);
        std::vector<uint32_t> vs(sample_n), ks(sample_n);
        for (std::size_t i = 0; i < sample_n; ++i) {
            std::size_t idx = (sample_n == n) ? i : ((i * n) / sample_n);
            vs[i] = tv_s[idx];
            ks[i] = tk_s[idx];
        }
        std::sort(vs.begin(), vs.end());
        std::sort(ks.begin(), ks.end());
        manual_v_bounds_.clear();
        manual_k_bounds_.clear();
        for (uint32_t b = 1; b < cfg_.grid_bins; ++b) {
            std::size_t idx = (static_cast<std::size_t>(b) * sample_n) / cfg_.grid_bins;
            if (idx >= sample_n) idx = sample_n - 1;
            manual_v_bounds_.push_back(vs[idx]);
            manual_k_bounds_.push_back(ks[idx]);
        }
        manual_min_tick_ = std::min(vs.front(), ks.front());
        manual_max_tick_ = std::max(vs.back(), ks.back());
        if (manual_max_tick_ <= manual_min_tick_) manual_max_tick_ = manual_min_tick_ + 86400u * 365u;
        has_manual_bounds_ = true;
    }

    // Streaming-native causal batch ingestion:
    // Processes records in commit-time (tk_s) order across self-sealing geometric MacroSegments.
    // Each MacroSegment calibrates valid-time splits strictly from already-arrived causal history
    // (or a tiny initial 2,048-record bootstrap prefix on Segment 0), with ZERO future look-ahead.
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
        if (n == 1) {
            insert_single_internal(ext_ids[0], vectors, tv_s[0], tv_e[0], tk_s[0], tk_e[0], true);
            return;
        }

        // Order batch arrivals by transaction commit timestamp tk_s (SQL:2011 SYSTEM_TIME monotonicity)
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        if (!std::is_sorted(tk_s, tk_s + n)) {
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return tk_s[a] < tk_s[b];
            });
        }

        std::size_t cursor = 0;
        while (cursor < n) {
            MacroSegment& seg = ensure_active_segment_for_stream(
                tv_s, tk_s, order.data() + cursor, n - cursor);

            std::size_t rem_in_seg = (seg.capacity > seg.num_nodes)
                ? static_cast<std::size_t>(seg.capacity - seg.num_nodes)
                : 0;
            if (rem_in_seg == 0) {
                seal_segment(seg);
                continue;
            }
            std::size_t take = std::min<std::size_t>(n - cursor, rem_in_seg);
            append_ordered_slice_to_segment(
                seg, ext_ids, vectors, tv_s, tv_e, tk_s, tk_e, order.data() + cursor, take, num_threads);
            cursor += take;

            if (seg.num_nodes >= seg.capacity || seg.active_gk >= cfg_.grid_bins) {
                seal_segment(seg);
            }
        }
    }

    // Online single-record streaming insert (O(log N) causal linking, automatic epoch & segment rollover)
    void insert_single(
        uint64_t ext_id,
        const float* vec,
        uint32_t tv_s, uint32_t tv_e,
        uint32_t tk_s, uint32_t tk_e) {
        insert_single_internal(ext_id, vec, tv_s, tv_e, tk_s, tk_e, true);
    }

    // O(1) Zero-Rewiring EVOLVE right-endpoint update (closes valid_end)
    bool update_valid_end(uint64_t ext_id, uint32_t new_tv_e) {
        return update_valid_end_internal(ext_id, new_tv_e, true);
    }

    // O(1) Zero-Rewiring RESTATE right-endpoint update (closes tx_end)
    bool update_tx_end(uint64_t ext_id, uint32_t new_tk_e) {
        return update_tx_end_internal(ext_id, new_tk_e, true);
    }

    // Bi-Temporal Point-in-Time Top-K Search with Selectivity-Aware Hybrid Cost Router
    QueryResult search(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k,
        uint32_t ef_search = 64,
        int force_mode = -1 /* -1: auto router, 0: force graph, 1: force scan */) const {
        auto t0 = std::chrono::steady_clock::now();
        QueryResult out;
        if (total_nodes_ == 0 || k == 0) return out;

        struct QualCellRef {
            const MacroSegment* seg;
            const EpochCell* cell;
            bool fully_born;
            uint64_t est_alive;
        };
        struct SegWork {
            const MacroSegment* seg;
            uint8_t q_gv;
            uint8_t q_gk;
            std::size_t cell_begin;
            std::size_t cell_end;
            uint64_t total_slots;
            uint64_t est_born;
            uint64_t est_alive;
        };

        std::vector<QualCellRef> qual_cells;
        qual_cells.reserve(segments_.size() * 64);
        std::vector<SegWork> seg_works;
        seg_works.reserve(segments_.size());

        uint64_t total_slots_all = 0;
        uint64_t est_born_total = 0;
        uint64_t est_alive_total = 0;

        // Step 1: Hierarchical Segment + Cell Bounding-Box & Mortality Histogram Pruning
        for (const auto& seg_ptr : segments_) {
            const MacroSegment& seg = *seg_ptr;
            if (seg.num_nodes == 0) continue;
            if (seg.min_tv_s > tv_q || seg.min_tk_s > tk_q) continue;

            uint8_t q_gv = bin_v_seg(seg, tv_q);
            uint8_t q_gk = bin_k_seg(seg, tk_q);

            // Accumulate causal-prefix born volume across ALL segments overlapping (-\infty, tv_q] x (-\infty, tk_q]
            // BEFORE segment/cell death pruning so survivor density is invariant to MacroSegment rollover boundaries!
            uint64_t seg_born = 0;
            for (uint8_t gk = 0; gk <= q_gk; ++gk) {
                if (seg.k_min_per_col[gk] > tk_q) continue;
                for (uint8_t gv = 0; gv <= q_gv; ++gv) {
                    const auto& cell = seg.cells[gk * cfg_.grid_bins + gv];
                    const std::size_t c_sz = cell.size();
                    if (c_sz == 0) continue;
                    if (cell.min_tv_s > tv_q || cell.min_tk_s > tk_q) continue;
                    bool fully_born = (cell.max_tv_s <= tv_q) && (cell.max_tk_s <= tk_q);
                    double birth_frac = 1.0;
                    if (!fully_born) {
                        double fv = (cell.max_tv_s > cell.min_tv_s)
                            ? static_cast<double>(tv_q - cell.min_tv_s + 1) / static_cast<double>(cell.max_tv_s - cell.min_tv_s + 1)
                            : 1.0;
                        double fk = (cell.max_tk_s > cell.min_tk_s)
                            ? static_cast<double>(tk_q - cell.min_tk_s + 1) / static_cast<double>(cell.max_tk_s - cell.min_tk_s + 1)
                            : 1.0;
                        birth_frac = std::clamp(fv, 0.05, 1.0) * std::clamp(fk, 0.05, 1.0);
                    }
                    seg_born += static_cast<uint64_t>(std::ceil(birth_frac * static_cast<double>(c_sz)));
                }
            }
            est_born_total += seg_born;

            if (seg.max_tv_e <= tv_q || seg.max_tk_e <= tk_q) continue;

            uint32_t q_mv = mort_bin_query_seg(seg, tv_q);
            uint32_t q_mk = mort_bin_query_seg(seg, tk_q);

            std::size_t c_begin = qual_cells.size();
            uint64_t seg_slots = 0;
            uint64_t seg_alive = 0;

            for (uint8_t gk = 0; gk <= q_gk; ++gk) {
                if (seg.k_min_per_col[gk] > tk_q) continue;
                for (uint8_t gv = 0; gv <= q_gv; ++gv) {
                    const auto& cell = seg.cells[gk * cfg_.grid_bins + gv];
                    const std::size_t c_sz = cell.size();
                    if (c_sz == 0) continue;
                    if (cell.min_tv_s > tv_q || cell.min_tk_s > tk_q) continue;
                    if (cell.max_tv_e <= tv_q || cell.max_tk_e <= tk_q) continue;

                    uint32_t sv = 0, sk = 0;
                    for (uint32_t b = q_mv; b < MORT_BINS; ++b) sv += cell.tv_end_hist[b];
                    for (uint32_t b = q_mk; b < MORT_BINS; ++b) sk += cell.tk_end_hist[b];
                    if (sv == 0 || sk == 0) continue;

                    const bool fully_born = (cell.max_tv_s <= tv_q) && (cell.max_tk_s <= tk_q);
                    const std::size_t n_blocks8 = (c_sz + 7u) / 8u;
                    const std::size_t sample_blocks = (force_mode == 1)
                        ? 0u
                        : std::min<std::size_t>(6u, n_blocks8);
                    uint32_t pilot_hits = 0;
                    uint32_t pilot_slots = 0;
                    for (std::size_t taken = 0; taken < sample_blocks; ++taken) {
                        std::size_t bi = (sample_blocks == n_blocks8)
                            ? taken
                            : std::min<std::size_t>(
                                n_blocks8 - 1,
                                (taken * n_blocks8 + (n_blocks8 / (2 * sample_blocks))) / sample_blocks);
                        std::size_t base = bi * 8u;
                        uint8_t mask = fully_born
                            ? simd::filter8_death_only(cell.tv_e.data + base, cell.tk_e.data + base, tv_q, tk_q)
                            : simd::filter8_full_bitemp(
                                cell.tv_s.data + base, cell.tv_e.data + base,
                                cell.tk_s.data + base, cell.tk_e.data + base, tv_q, tk_q);
                        uint32_t valid_lanes = static_cast<uint32_t>(std::min<std::size_t>(8u, c_sz - base));
                        if (valid_lanes < 8u) mask &= static_cast<uint8_t>((1u << valid_lanes) - 1u);
                        pilot_hits += static_cast<uint32_t>(__builtin_popcount(mask));
                        pilot_slots += valid_lanes;
                    }

                    uint64_t est_alive = (pilot_slots > 0)
                        ? static_cast<uint64_t>(std::llround((static_cast<double>(pilot_hits) / static_cast<double>(pilot_slots)) * static_cast<double>(c_sz)))
                        : 0ULL;

                    qual_cells.push_back({&seg, &cell, fully_born, est_alive});
                    seg_slots += c_sz;
                    seg_alive += est_alive;
                }
            }
            if (qual_cells.size() > c_begin) {
                seg_works.push_back({&seg, q_gv, q_gk, c_begin, qual_cells.size(), seg_slots, seg_born, seg_alive});
                total_slots_all += seg_slots;
                est_alive_total += seg_alive;
            }
        }

        out.stats.visited_cells = qual_cells.size();
        if (qual_cells.empty()) {
            auto t1 = std::chrono::steady_clock::now();
            out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            return out;
        }

        // Order qualifying segments largest-alive-first for analytical cost modeling & multi-segment execution
        std::sort(seg_works.begin(), seg_works.end(), [](const SegWork& a, const SegWork& b) {
            return a.est_alive > b.est_alive;
        });

        simd::SQ8Query sq_q = simd::prepare_sq8_query(q_vec, cfg_.dim);
        float surv_ratio = static_cast<float>(est_alive_total) / static_cast<float>(std::max<uint64_t>(1, est_born_total));

        auto compute_seg_beam = [&](std::size_t sw_i, uint64_t seg_alive) -> uint32_t {
            const uint32_t hi_ef = std::max<uint32_t>(ef_search, k * 2);
            if (sw_i == 0) return hi_ef;
            double alive_share = (est_alive_total > 0)
                ? static_cast<double>(seg_alive) / static_cast<double>(est_alive_total)
                : (1.0 / static_cast<double>(seg_works.size()));
            // Concave sqrt allocation preserves recall across historical MacroSegments without starving older epochs
            double seg_w = std::max(0.55, std::sqrt(alive_share));
            const uint32_t lo_ef = std::min<uint32_t>(hi_ef, std::max<uint32_t>(k * 3, 32u));
            return std::clamp<uint32_t>(
                static_cast<uint32_t>(std::ceil(static_cast<double>(ef_search) * seg_w)),
                lo_ef,
                hi_ef);
        };

        const double dim_scale = static_cast<double>(cfg_.dim) / 768.0;
        // Working-set LLC-to-DRAM latency scale (auto-calibratable via calibrate_router_hardware)
        const double eta_mem = std::clamp(
            static_cast<double>(total_nodes_) / std::max(1.0, cfg_.router_llc_knee), 0.24, 1.0);
        const double kGamma0 = cfg_.router_gamma_0;
        const double kGammaTs = cfg_.router_gamma_ts;
        const double kGammaHub = cfg_.router_gamma_hub;
        const double kGammaLlc = cfg_.router_gamma_llc;
        const double kGammaDim = cfg_.router_gamma_dim;

        auto est_seg_scan_cost = [dim_scale, kGamma0, kGammaTs](uint64_t slots, uint64_t alive) -> double {
            return (0.5 * kGamma0) + kGammaTs * (static_cast<double>(slots) / 8.0) + dim_scale * static_cast<double>(alive);
        };
        auto est_seg_graph_cost = [dim_scale, eta_mem, kGammaHub, kGammaLlc, kGammaDim](
            uint32_t beam, uint64_t born, uint64_t alive) -> double {
            double rho = std::max(0.025, static_cast<double>(alive) / static_cast<double>(std::max<uint64_t>(1, born)));
            return eta_mem * (kGammaHub + ((kGammaLlc + kGammaDim * dim_scale) * static_cast<double>(beam)) / std::pow(rho, 0.80));
        };

        bool use_scan = false;
        if (force_mode == 1) {
            use_scan = true;
        } else if (force_mode == 0) {
            use_scan = false;
        } else {
            // Analytical Multi-Segment Cost Router (normalized to one 768-d maddubs evaluation; zero hardcoded cap!):
            // Compares contiguous cell-pruned AVX2 SQ8 scan cost against multi-segment causal graph traversal cost
            double c_scan_all = kGamma0 + kGammaTs * (static_cast<double>(total_slots_all) / 8.0)
                              + dim_scale * static_cast<double>(est_alive_total);
            double c_hybrid_graph = 0.0;
            bool any_graph_seg = false;
            for (std::size_t sw_i = 0; sw_i < seg_works.size(); ++sw_i) {
                const auto& sw = seg_works[sw_i];
                uint32_t sb = compute_seg_beam(sw_i, sw.est_alive);
                double cs = est_seg_scan_cost(sw.total_slots, sw.est_alive);
                double cg = est_seg_graph_cost(sb, sw.est_born, sw.est_alive);
                if (sw_i == 0 || cg < cs) {
                    c_hybrid_graph += cg;
                    if (cg < cs) any_graph_seg = true;
                } else {
                    c_hybrid_graph += cs;
                }
            }
            if (!any_graph_seg || (0.92 * c_scan_all <= c_hybrid_graph) || surv_ratio < cfg_.min_graph_survivor_ratio) {
                use_scan = true;
            }
        }

        // Global candidate handle: (seg_id, local_id) packed into uint64_t
        // Full-scan path needs only max(6k, 64) SQ8 finalists (matching Grid-SQ8-Scan), whereas multi-segment graph uses min(3*ef, 1536)!
        using GlobalCand = std::pair<float, uint64_t>;
        const uint32_t scan_rerank_cap = (force_mode == 1)
            ? std::max<uint32_t>(k, ef_search)
            : std::max<uint32_t>(k * 6, 64u);
        const uint32_t rerank_cap = std::max<uint32_t>(k * 8, std::min<uint32_t>(ef_search * 3, 1536u));

        if (use_scan) {
            out.stats.scan_cells = qual_cells.size();
            std::priority_queue<GlobalCand> sq_top;
            for (const auto& qc : qual_cells) {
                const MacroSegment& seg = *qc.seg;
                const EpochCell& cell = *qc.cell;
                const std::size_t sz = cell.size();
                const uint32_t* tvs = cell.tv_s.data;
                const uint32_t* tve = cell.tv_e.data;
                const uint32_t* tks = cell.tk_s.data;
                const uint32_t* tke = cell.tk_e.data;
                const bool fb = qc.fully_born;
                const uint64_t seg_prefix = static_cast<uint64_t>(seg.seg_id) << 32;

                for (std::size_t base = 0; base < sz; base += 8) {
                    uint8_t mask = fb
                        ? simd::filter8_death_only(tve + base, tke + base, tv_q, tk_q)
                        : simd::filter8_full_bitemp(tvs + base, tve + base, tks + base, tke + base, tv_q, tk_q);
                    out.stats.predicate_checks += 8;
                    while (mask != 0) {
                        unsigned bit = static_cast<unsigned>(__builtin_ctz(mask));
                        mask &= static_cast<uint8_t>(mask - 1);
                        std::size_t slot = base + bit;
                        if (slot >= sz) break;
                        float adist = simd::sq8_approx_l2_sqr(
                            sq_q,
                            cell.sq8_codes.data + slot * padded_dim_,
                            cell.sq8_meta[slot]);
                        out.stats.distance_computations++;
                        uint64_t gh = seg_prefix | static_cast<uint64_t>(cell.local_ids[slot]);
                        if (sq_top.size() < scan_rerank_cap) {
                            sq_top.emplace(adist, gh);
                        } else if (adist < sq_top.top().first) {
                            sq_top.pop();
                            sq_top.emplace(adist, gh);
                        }
                    }
                }
            }
            finalize_exact_rerank_global(q_vec, k, sq_top, out);
        } else {
            std::priority_queue<GlobalCand> global_valid_top;
            thread_local std::vector<uint32_t> tl_visited;
            thread_local uint32_t tl_token = 1;

            for (std::size_t sw_i = 0; sw_i < seg_works.size(); ++sw_i) {
                const SegWork& sw = seg_works[sw_i];
                const MacroSegment& seg = *sw.seg;
                const uint64_t seg_prefix = static_cast<uint64_t>(seg.seg_id) << 32;
                std::size_t n_seg_cells = sw.cell_end - sw.cell_begin;
                uint32_t seg_beam = compute_seg_beam(sw_i, sw.est_alive);

                // Per-segment hybrid sub-router: scan small/sparse secondary segments directly when cheaper than graph
                bool scan_this_seg = (force_mode != 0) && (sw_i > 0)
                    && (est_seg_scan_cost(sw.total_slots, sw.est_alive) <= est_seg_graph_cost(seg_beam, sw.est_born, sw.est_alive));
                if (scan_this_seg) {
                    out.stats.scan_cells += n_seg_cells;
                    for (std::size_t ci = sw.cell_begin; ci < sw.cell_end; ++ci) {
                        const EpochCell& cell = *qual_cells[ci].cell;
                        const std::size_t sz = cell.size();
                        const bool fb = qual_cells[ci].fully_born;
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
                                float adist = simd::sq8_approx_l2_sqr(
                                    sq_q, cell.sq8_codes.data + slot * padded_dim_, cell.sq8_meta[slot]);
                                out.stats.distance_computations++;
                                uint64_t gh = seg_prefix | static_cast<uint64_t>(cell.local_ids[slot]);
                                if (global_valid_top.size() < rerank_cap) {
                                    global_valid_top.emplace(adist, gh);
                                } else if (adist < global_valid_top.top().first) {
                                    global_valid_top.pop();
                                    global_valid_top.emplace(adist, gh);
                                }
                            }
                        }
                    }
                    continue;
                }

                out.stats.graph_cells += n_seg_cells;
                if (tl_visited.size() < seg.num_nodes) tl_visited.assign(seg.num_nodes, 0);

                float seg_surv = static_cast<float>(sw.est_alive) / static_cast<float>(std::max<uint64_t>(1, sw.est_born));
                const uint32_t bridge_cap = seg_beam * 2 + 16;

                using LocalPair = std::pair<float, uint32_t>;
                std::priority_queue<LocalPair, std::vector<LocalPair>, std::greater<LocalPair>> cand_min;
                std::priority_queue<LocalPair> beam_bound;
                std::priority_queue<LocalPair> seg_valid_top;

                // Stage A: Hierarchical Layer-1 Causal Highway Zoom-In (uses dedicated u_tok so Layer-0 never skips unseeded Layer-1 nodes!)
                std::vector<LocalPair> u_best;
                if (!seg.upper_hubs.empty()) {
                    if (++tl_token == 0) {
                        std::fill(tl_visited.begin(), tl_visited.end(), 0);
                        tl_token = 1;
                    }
                    const uint32_t u_tok = tl_token;
                    std::priority_queue<LocalPair, std::vector<LocalPair>, std::greater<LocalPair>> u_cand;
                    std::priority_queue<LocalPair> u_top;
                    const uint32_t u_beam = 16;
                    std::size_t u_stride = std::max<std::size_t>(1, seg.upper_hubs.size() / 24);
                    for (std::size_t ui = 0; ui < seg.upper_hubs.size(); ui += u_stride) {
                        uint32_t uh = seg.upper_hubs[ui];
                        if (seg.node_gv[uh] > sw.q_gv || seg.node_gk[uh] > sw.q_gk) continue;
                        tl_visited[uh] = u_tok;
                        float d = node_sq8_dist(seg, uh, sq_q);
                        out.stats.distance_computations++;
                        u_cand.emplace(d, uh);
                        u_top.emplace(d, uh);
                        if (u_top.size() > u_beam) u_top.pop();
                    }
                    uint32_t u_exp = 0;
                    while (!u_cand.empty() && u_exp < 48) {
                        auto [ud, ucurr] = u_cand.top();
                        u_cand.pop();
                        if (u_top.size() >= u_beam && ud > u_top.top().first) break;
                        u_exp++;
                        if (seg.adj_upper[ucurr].empty()) continue;
                        for (uint32_t packed : seg.adj_upper[ucurr][0]) {
                            if (unpack_gv(packed) > sw.q_gv || unpack_gk(packed) > sw.q_gk) continue;
                            uint32_t unb = unpack_id(packed);
                            if (tl_visited[unb] == u_tok) continue;
                            tl_visited[unb] = u_tok;
                            float d = node_sq8_dist(seg, unb, sq_q);
                            out.stats.distance_computations++;
                            if (u_top.size() < u_beam || d < u_top.top().first) {
                                u_cand.emplace(d, unb);
                                u_top.emplace(d, unb);
                                if (u_top.size() > u_beam) u_top.pop();
                            }
                        }
                    }
                    u_best.reserve(u_top.size());
                    while (!u_top.empty()) { u_best.push_back(u_top.top()); u_top.pop(); }
                    std::sort(u_best.begin(), u_best.end());
                }

                // Fresh Layer-0 visited token!
                if (++tl_token == 0) {
                    std::fill(tl_visited.begin(), tl_visited.end(), 0);
                    tl_token = 1;
                }
                const uint32_t tok = tl_token;
                std::size_t keep_u = std::min<std::size_t>(8, u_best.size());
                for (std::size_t bi = 0; bi < keep_u; ++bi) {
                    auto [hd, hid] = u_best[bi];
                    tl_visited[hid] = tok;
                    cand_min.emplace(hd, hid);
                    beam_bound.emplace(hd, hid);
                    out.stats.predicate_checks++;
                    if (bitemp_visible(seg.g_tv_s[hid], seg.g_tv_e[hid], seg.g_tk_s[hid], seg.g_tk_e[hid], tv_q, tk_q)) {
                        seg_valid_top.emplace(hd, hid);
                        if (seg_valid_top.size() > seg_beam) seg_valid_top.pop();
                    }
                }

                // Stage B: Compact stratified live cell seeds across qualifying cells
                const uint32_t max_seeds = (seg_surv < 0.15f) ? 32u : 16u;
                uint32_t seeded = 0;
                std::size_t cell_stride = std::max<std::size_t>(1, n_seg_cells / max_seeds);
                for (std::size_t ci = sw.cell_end; ci > sw.cell_begin && seeded < max_seeds;) {
                    std::size_t idx_c = ci - 1;
                    const EpochCell& cell = *qual_cells[idx_c].cell;
                    for (uint32_t ep : cell.entry_nodes) {
                        if (tl_visited[ep] == tok) continue;
                        out.stats.predicate_checks++;
                        bool ep_alive = bitemp_visible(seg.g_tv_s[ep], seg.g_tv_e[ep], seg.g_tk_s[ep], seg.g_tk_e[ep], tv_q, tk_q);
                        if (!ep_alive && seg_surv >= 0.10f) continue;
                        tl_visited[ep] = tok;
                        float d = node_sq8_dist(seg, ep, sq_q);
                        out.stats.distance_computations++;
                        cand_min.emplace(d, ep);
                        beam_bound.emplace(d, ep);
                        if (beam_bound.size() > bridge_cap) beam_bound.pop();
                        if (ep_alive) {
                            seg_valid_top.emplace(d, ep);
                            if (seg_valid_top.size() > seg_beam) seg_valid_top.pop();
                        }
                        seeded++;
                        break;
                    }
                    if (ci - sw.cell_begin <= cell_stride) break;
                    ci -= cell_stride;
                }

                const uint64_t max_expansions = static_cast<uint64_t>(seg_beam) * 16ULL;
                uint64_t expansions = 0;
                const uint32_t live_warm_min = std::min<uint32_t>(seg_beam / 2, std::max<uint32_t>(k, 16u));

                while (!cand_min.empty() && expansions < max_expansions) {
                    auto [cdist, curr] = cand_min.top();
                    cand_min.pop();
                    if (seg_valid_top.size() >= seg_beam && cdist > seg_valid_top.top().first) {
                        break;
                    }
                    expansions++;

                    // 1) Traverse Causal-Prefix neighbors (guaranteed gv <= q_gv && gk <= q_gk by construction!)
                    const auto& pre_list = seg.adj_pre[curr];
                    for (uint32_t packed : pre_list) {
                        uint32_t nb = unpack_id(packed);
                        if (tl_visited[nb] == tok) continue;
                        tl_visited[nb] = tok;

                        out.stats.predicate_checks++;
                        bool alive = bitemp_visible(seg.g_tv_s[nb], seg.g_tv_e[nb], seg.g_tk_s[nb], seg.g_tk_e[nb], tv_q, tk_q);
                        // Dead-node bandwidth bypass: when live subgraph is well above percolation (surv >= 0.45)
                        // and live beam is warmed up, skip 768-byte SQ8 vector fetch on expired tuples!
                        if (!alive && seg_surv >= 0.45f && seg_valid_top.size() >= live_warm_min) {
                            continue;
                        }

                        float d = node_sq8_dist(seg, nb, sq_q);
                        out.stats.distance_computations++;

                        if (alive) {
                            if (seg_valid_top.size() < seg_beam) {
                                seg_valid_top.emplace(d, nb);
                            } else if (d < seg_valid_top.top().first) {
                                seg_valid_top.pop();
                                seg_valid_top.emplace(d, nb);
                            }
                        }
                        bool accept_bridge = (seg_valid_top.size() < seg_beam)
                            ? (beam_bound.size() < bridge_cap || d < beam_bound.top().first)
                            : (d < seg_valid_top.top().first);
                        if (accept_bridge) {
                            cand_min.emplace(d, nb);
                            beam_bound.emplace(d, nb);
                            if (beam_bound.size() > bridge_cap) beam_bound.pop();
                        }
                    }

                    // 2) Traverse Forward-Bridge neighbors ONLY if inline 8-bit birth tag lies inside [0..q_gv] x [0..q_gk]
                    const auto& fwd_list = seg.adj_fwd[curr];
                    for (uint32_t packed : fwd_list) {
                        if (unpack_gv(packed) > sw.q_gv || unpack_gk(packed) > sw.q_gk) continue; // 1-cycle ALU tag prune!
                        uint32_t nb = unpack_id(packed);
                        if (tl_visited[nb] == tok) continue;
                        tl_visited[nb] = tok;

                        out.stats.predicate_checks++;
                        bool alive = bitemp_visible(seg.g_tv_s[nb], seg.g_tv_e[nb], seg.g_tk_s[nb], seg.g_tk_e[nb], tv_q, tk_q);
                        if (!alive && seg_surv >= 0.45f && seg_valid_top.size() >= live_warm_min) {
                            continue;
                        }

                        float d = node_sq8_dist(seg, nb, sq_q);
                        out.stats.distance_computations++;

                        if (alive) {
                            if (seg_valid_top.size() < seg_beam) {
                                seg_valid_top.emplace(d, nb);
                            } else if (d < seg_valid_top.top().first) {
                                seg_valid_top.pop();
                                seg_valid_top.emplace(d, nb);
                            }
                        }
                        bool accept_bridge = (seg_valid_top.size() < seg_beam)
                            ? (beam_bound.size() < bridge_cap || d < beam_bound.top().first)
                            : (d < seg_valid_top.top().first);
                        if (accept_bridge) {
                            cand_min.emplace(d, nb);
                            beam_bound.emplace(d, nb);
                            if (beam_bound.size() > bridge_cap) beam_bound.pop();
                        }
                    }
                }

                while (!seg_valid_top.empty()) {
                    auto [sd, lid] = seg_valid_top.top();
                    seg_valid_top.pop();
                    uint64_t gh = seg_prefix | static_cast<uint64_t>(lid);
                    if (global_valid_top.size() < rerank_cap) {
                        global_valid_top.emplace(sd, gh);
                    } else if (sd < global_valid_top.top().first) {
                        global_valid_top.pop();
                        global_valid_top.emplace(sd, gh);
                    }
                }
            }

            if (global_valid_top.size() < k && force_mode != 0) {
                QueryResult fb_res = search(q_vec, tv_q, tk_q, k, ef_search, 1);
                fb_res.stats.distance_computations += out.stats.distance_computations;
                fb_res.stats.predicate_checks += out.stats.predicate_checks;
                fb_res.stats.graph_cells += out.stats.graph_cells;
                auto t1_fb = std::chrono::steady_clock::now();
                fb_res.stats.latency_us = std::chrono::duration<double, std::micro>(t1_fb - t0).count();
                return fb_res;
            }
            finalize_exact_rerank_global(q_vec, k, global_valid_top, out);
        }

        auto t1 = std::chrono::steady_clock::now();
        out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }

    // Exact brute-force Ground Truth search across all segments
    QueryResult exact_search(
        const float* q_vec,
        uint32_t tv_q,
        uint32_t tk_q,
        uint32_t k) const {
        auto t0 = std::chrono::steady_clock::now();
        QueryResult out;
        using Pair = std::pair<float, uint64_t>;
        std::priority_queue<Pair> topk;
        for (const auto& seg_ptr : segments_) {
            const MacroSegment& seg = *seg_ptr;
            for (std::size_t i = 0; i < seg.num_nodes; ++i) {
                if (bitemp_visible(seg.g_tv_s[i], seg.g_tv_e[i], seg.g_tk_s[i], seg.g_tk_e[i], tv_q, tk_q)) {
                    float d = simd::l2_sqr_fp32(q_vec, seg.fp32_ptr(i, cfg_.dim), cfg_.dim);
                    out.stats.distance_computations++;
                    if (topk.size() < k) {
                        topk.emplace(d, seg.ext_ids[i]);
                    } else if (d < topk.top().first) {
                        topk.pop();
                        topk.emplace(d, seg.ext_ids[i]);
                    }
                }
            }
        }
        out.ids.resize(topk.size());
        out.distances.resize(topk.size());
        for (std::size_t idx = topk.size(); idx > 0; --idx) {
            out.ids[idx - 1] = topk.top().second;
            out.distances[idx - 1] = topk.top().first;
            topk.pop();
        }
        auto t1 = std::chrono::steady_clock::now();
        out.stats.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return out;
    }

    // Retrieve exact FP32 vector pointer for an external ID (works transparently over RAM or NVMe mmap!)
    const float* get_vector_ptr(uint64_t ext_id) const noexcept {
        auto it = ext_to_loc_.find(ext_id);
        if (it == ext_to_loc_.end()) return nullptr;
        uint32_t sid = static_cast<uint32_t>(it->second >> 32);
        uint32_t lid = static_cast<uint32_t>(it->second & 0xFFFFFFFFu);
        if (sid >= segments_.size() || lid >= segments_[sid]->num_nodes) return nullptr;
        return segments_[sid]->fp32_ptr(lid, cfg_.dim);
    }

    std::size_t size() const noexcept { return total_nodes_; }
    std::size_t num_segments() const noexcept { return segments_.size(); }

    // Counted core index array bytes (excludes global_dir_ hash map & NVMe-mmap cold FP32 vectors once offloaded)
    std::size_t memory_bytes() const noexcept {
        std::size_t bytes = 0;
        for (const auto& sp : segments_) {
            const MacroSegment& seg = *sp;
            bytes += seg.ext_ids.size() * sizeof(uint64_t)
                + seg.node_gv.size() * sizeof(uint8_t)
                + seg.node_gk.size() * sizeof(uint8_t)
                + seg.node_cid.size() * sizeof(uint8_t)
                + seg.node_cell_slot.size() * sizeof(uint32_t)
                + seg.node_level.size() * sizeof(uint8_t)
                + (seg.g_tv_s.size + seg.g_tv_e.size + seg.g_tk_s.size + seg.g_tk_e.size) * sizeof(uint32_t)
                + seg.fp32_vecs.size * sizeof(float);
            for (const auto& v : seg.adj_pre) bytes += v.capacity() * sizeof(uint32_t);
            for (const auto& v : seg.adj_fwd) bytes += v.capacity() * sizeof(uint32_t);
            for (const auto& lvls : seg.adj_upper) {
                for (const auto& u : lvls) bytes += u.capacity() * sizeof(uint32_t);
            }
            for (const auto& c : seg.cells) {
                bytes += (c.tv_s.size + c.tv_e.size + c.tk_s.size + c.tk_e.size) * sizeof(uint32_t);
                bytes += c.sq8_codes.size * sizeof(int8_t);
                bytes += c.sq8_meta.size() * sizeof(simd::SQ8VectorMeta);
                bytes += c.local_ids.size() * sizeof(uint32_t);
            }
        }
        return bytes;
    }

    // Complete logical heap footprint including ext_to_loc_ unordered_map buckets/nodes, std::vector headers, and locks
    std::size_t total_ram_bytes() const noexcept {
        std::size_t bytes = memory_bytes();
        // std::unordered_map<uint64_t, uint64_t> bucket array + libstdc++ _Hash_node<pair<const uint64_t, uint64_t>, false> (24B + 16B malloc header)
        bytes += ext_to_loc_.bucket_count() * sizeof(void*)
               + ext_to_loc_.size() * (sizeof(void*) + 2 * sizeof(uint64_t) + 16u);
        bytes += causal_lag_reservoir_.capacity() * sizeof(uint32_t);
        for (const auto& sp : segments_) {
            const MacroSegment& seg = *sp;
            bytes += sizeof(MacroSegment)
                   + seg.adj_pre.capacity() * sizeof(std::vector<uint32_t>)
                   + seg.adj_fwd.capacity() * sizeof(std::vector<uint32_t>)
                   + seg.adj_upper.capacity() * sizeof(std::vector<std::vector<uint32_t>>)
                   + seg.node_locks.size() * sizeof(SpinLock)
                   + seg.upper_hubs.capacity() * sizeof(uint32_t)
                   + seg.cells.capacity() * sizeof(decltype(seg.cells)::value_type);
        }
        return bytes;
    }

    // Exact logical RAM footprint excluding HNSW graph adjacency lists & hubs (for controlled BiT-ScanOnly ablation)
    std::size_t non_graph_ram_bytes() const noexcept {
        std::size_t tot = total_ram_bytes();
        std::size_t g_bytes = 0;
        for (const auto& sp : segments_) {
            const MacroSegment& seg = *sp;
            g_bytes += seg.adj_pre.capacity() * sizeof(std::vector<uint32_t>)
                     + seg.adj_fwd.capacity() * sizeof(std::vector<uint32_t>)
                     + seg.adj_upper.capacity() * sizeof(std::vector<std::vector<uint32_t>>)
                     + seg.upper_hubs.capacity() * sizeof(uint32_t);
            for (const auto& v : seg.adj_pre) g_bytes += v.capacity() * sizeof(uint32_t);
            for (const auto& v : seg.adj_fwd) g_bytes += v.capacity() * sizeof(uint32_t);
            for (const auto& lv : seg.adj_upper) {
                g_bytes += lv.capacity() * sizeof(std::vector<uint32_t>);
                for (const auto& v : lv) g_bytes += v.capacity() * sizeof(uint32_t);
            }
        }
        return (tot > g_bytes) ? (tot - g_bytes) : tot;
    }

    bool verify_identical_state(const BiTHNSWIndex& other) const noexcept {
        if (total_nodes_ != other.total_nodes_ || segments_.size() != other.segments_.size()) return false;
        if (ext_to_loc_.size() != other.ext_to_loc_.size()) return false;
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            const MacroSegment& a = *segments_[s];
            const MacroSegment& b = *other.segments_[s];
            if (a.num_nodes != b.num_nodes) return false;
            for (std::size_t i = 0; i < a.num_nodes; ++i) {
                if (a.ext_ids[i] != b.ext_ids[i] ||
                    a.g_tv_s[i] != b.g_tv_s[i] || a.g_tv_e[i] != b.g_tv_e[i] ||
                    a.g_tk_s[i] != b.g_tk_s[i] || a.g_tk_e[i] != b.g_tk_e[i]) {
                    return false;
                }
                const float* va = a.fp32_ptr(i, cfg_.dim);
                const float* vb = b.fp32_ptr(i, cfg_.dim);
                if (std::memcmp(va, vb, cfg_.dim * sizeof(float)) != 0) return false;
            }
        }
        return true;
    }

    std::size_t mmap_bytes() const noexcept {
        std::size_t bytes = 0;
        for (const auto& sp : segments_) {
            bytes += sp->fp32_mmap_bytes;
        }
        return bytes;
    }

    // Offload cold master FP32 vectors (3,072 B/rec) to NVMe via POSIX mmap, retaining 1,033 B/rec core index in RAM
    PersistenceStats offload_fp32_to_mmap(
        const std::string& dir_path,
        bool include_active = true,
        bool drop_os_cache = false) {
        auto t0 = std::chrono::steady_clock::now();
        PersistenceStats st{};
        #if defined(__unix__) || defined(__APPLE__)
        ::mkdir(dir_path.c_str(), 0755);
        for (auto& sp : segments_) {
            MacroSegment& seg = *sp;
            if (seg.num_nodes == 0 || seg.fp32_mmap_ptr != nullptr) continue;
            if (!include_active && !seg.sealed) continue;

            std::size_t bytes = seg.num_nodes * cfg_.dim * sizeof(float);
            std::string fpath = dir_path + "/seg_" + std::to_string(seg.seg_id) + ".fp32";
            int wfd = ::open(fpath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            if (wfd < 0) throw std::runtime_error("Failed to create NVMe FP32 offload file: " + fpath);

            const uint8_t* src = reinterpret_cast<const uint8_t*>(seg.fp32_vecs.data);
            std::size_t rem = bytes;
            while (rem > 0) {
                ssize_t w = ::write(wfd, src, std::min<std::size_t>(rem, 8 * 1024 * 1024));
                if (w <= 0) {
                    ::close(wfd);
                    throw std::runtime_error("Write failed during NVMe FP32 offload: " + fpath);
                }
                src += static_cast<std::size_t>(w);
                rem -= static_cast<std::size_t>(w);
            }
            ::fdatasync(wfd);
            #ifdef POSIX_FADV_DONTNEED
            if (drop_os_cache) {
                ::posix_fadvise(wfd, 0, static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
            }
            #endif
            ::close(wfd);

            int rfd = ::open(fpath.c_str(), O_RDONLY);
            if (rfd < 0) throw std::runtime_error("Failed to open NVMe FP32 offload file for mmap: " + fpath);
            void* mapped = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, rfd, 0);
            if (mapped == MAP_FAILED) {
                ::close(rfd);
                throw std::runtime_error("mmap failed on NVMe FP32 offload file: " + fpath);
            }
            #ifdef MADV_RANDOM
            ::madvise(mapped, bytes, MADV_RANDOM);
            #endif
            #ifdef MADV_DONTNEED
            if (drop_os_cache) {
                ::madvise(mapped, bytes, MADV_DONTNEED);
            }
            #endif

            seg.fp32_mmap_fd = rfd;
            seg.fp32_mmap_ptr = static_cast<float*>(mapped);
            seg.fp32_mmap_bytes = bytes;
            seg.fp32_mmap_path = fpath;
            seg.sealed = true; // Offloaded segments are immutable; future streaming writes open a fresh RAM segment
            seg.fp32_vecs.release();

            st.bytes_transferred += bytes;
            st.records_processed += seg.num_nodes;
        }
        #endif
        auto t1 = std::chrono::steady_clock::now();
        st.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        st.throughput_mb_s = (st.elapsed_ms > 0.0)
            ? (static_cast<double>(st.bytes_transferred) / (1024.0 * 1024.0)) / (st.elapsed_ms / 1000.0)
            : 0.0;
        return st;
    }

    void drop_fp32_mmap_page_cache() noexcept {
        #if defined(__unix__) || defined(__APPLE__)
        for (auto& sp : segments_) {
            MacroSegment& seg = *sp;
            if (seg.fp32_mmap_ptr != nullptr && seg.fp32_mmap_bytes > 0 && seg.fp32_mmap_fd >= 0) {
                // Unmap BEFORE posix_fadvise(DONTNEED) so Linux unconditionally evicts clean file-backed pages,
                // then re-establish the read-only MADV_RANDOM mapping!
                ::munmap(seg.fp32_mmap_ptr, seg.fp32_mmap_bytes);
                seg.fp32_mmap_ptr = nullptr;
                #ifdef POSIX_FADV_DONTNEED
                ::posix_fadvise(seg.fp32_mmap_fd, 0, static_cast<off_t>(seg.fp32_mmap_bytes), POSIX_FADV_DONTNEED);
                #endif
                void* mapped = ::mmap(nullptr, seg.fp32_mmap_bytes, PROT_READ, MAP_SHARED, seg.fp32_mmap_fd, 0);
                if (mapped != MAP_FAILED) {
                    #ifdef MADV_RANDOM
                    ::madvise(mapped, seg.fp32_mmap_bytes, MADV_RANDOM);
                    #endif
                    #ifdef MADV_DONTNEED
                    ::madvise(mapped, seg.fp32_mmap_bytes, MADV_DONTNEED);
                    #endif
                    seg.fp32_mmap_ptr = static_cast<float*>(mapped);
                }
            }
        }
        #endif
    }

    // =========================================================================
    // NVMe Snapshot Checkpointing (mmap) + Append-Only Mutation WAL Recovery
    // =========================================================================
    #pragma pack(push, 1)
    struct SnapshotFileHeader {
        uint64_t magic = 0x5049544D534E5033ULL; // "PITMSNP3"
        uint32_t dim = 768;
        uint32_t padded_dim = 768;
        uint32_t grid_bins = 16;
        uint32_t m_pre = 16;
        uint32_t m_fwd = 12;
        uint32_t m_upper = 16;
        uint64_t total_nodes = 0;
        uint32_t num_segments = 0;
        uint64_t checkpoint_lsn = 0;
    };

    struct WALRecordHeader {
        uint32_t magic = 0x5057414Cu; // "PWAL"
        uint8_t op = 0;               // 1: INSERT, 2: UPDATE_VALID_END, 3: UPDATE_TX_END, 10: BEGIN_TX, 11: COMMIT_TX
        uint8_t reserved[3] = {0, 0, 0};
        uint64_t lsn = 0;             // Monotonic log sequence number
        uint64_t ext_id = 0;
        uint32_t tv_s = 0;
        uint32_t tv_e = 0;
        uint32_t tk_s = 0;
        uint32_t tk_e = 0;
        uint32_t payload_bytes = 0;
        uint32_t crc32 = 0;
    };
    #pragma pack(pop)

    PersistenceStats save_snapshot(const std::string& path, bool fsync_disk = true) const {
        auto t0 = std::chrono::steady_clock::now();
        std::string tmp_path = path + ".tmp." + std::to_string(static_cast<unsigned long long>(::getpid()));
        int fd = ::open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) throw std::runtime_error("Failed to open staging snapshot path for writing: " + tmp_path);

        std::vector<uint8_t> buf;
        buf.reserve(4 * 1024 * 1024);
        std::size_t total_written = 0;

        auto abort_tmp = [&](const char* msg) {
            ::close(fd);
            ::unlink(tmp_path.c_str());
            throw std::runtime_error(msg);
        };

        auto flush_buf = [&]() {
            const uint8_t* p = buf.data();
            std::size_t rem = buf.size();
            while (rem > 0) {
                ssize_t w = ::write(fd, p, rem);
                if (w <= 0) abort_tmp("Snapshot write failed");
                p += static_cast<std::size_t>(w);
                rem -= static_cast<std::size_t>(w);
                total_written += static_cast<std::size_t>(w);
            }
            buf.clear();
        };
        auto emit_bytes = [&](const void* src, std::size_t len) {
            if (len == 0) return;
            if (buf.size() + len > 4 * 1024 * 1024) flush_buf();
            if (len > 4 * 1024 * 1024) {
                const uint8_t* p = static_cast<const uint8_t*>(src);
                std::size_t rem = len;
                while (rem > 0) {
                    ssize_t w = ::write(fd, p, rem);
                    if (w <= 0) abort_tmp("Snapshot large write failed");
                    p += static_cast<std::size_t>(w);
                    rem -= static_cast<std::size_t>(w);
                    total_written += static_cast<std::size_t>(w);
                }
            } else {
                const uint8_t* p = static_cast<const uint8_t*>(src);
                buf.insert(buf.end(), p, p + len);
            }
        };
        auto emit_u64 = [&](uint64_t v) { emit_bytes(&v, sizeof(v)); };
        auto emit_u32 = [&](uint32_t v) { emit_bytes(&v, sizeof(v)); };

        SnapshotFileHeader hdr{};
        hdr.dim = static_cast<uint32_t>(cfg_.dim);
        hdr.padded_dim = static_cast<uint32_t>(padded_dim_);
        hdr.grid_bins = cfg_.grid_bins;
        hdr.m_pre = cfg_.m_pre;
        hdr.m_fwd = cfg_.m_fwd;
        hdr.m_upper = cfg_.m_upper;
        hdr.total_nodes = total_nodes_;
        hdr.num_segments = static_cast<uint32_t>(segments_.size());
        hdr.checkpoint_lsn = wal_lsn_;
        emit_bytes(&hdr, sizeof(hdr));

        for (const auto& sp : segments_) {
            const MacroSegment& seg = *sp;
            emit_u32(seg.seg_id);
            emit_u64(seg.capacity);
            emit_u64(seg.epoch_row_cap);
            uint8_t flags[3] = {static_cast<uint8_t>(seg.sealed ? 1 : 0),
                                static_cast<uint8_t>(seg.bounds_ready ? 1 : 0),
                                seg.active_gk};
            emit_bytes(flags, 3);
            uint32_t s_bounds[8] = {
                seg.min_tv_s, seg.max_tv_s, seg.min_tk_s, seg.max_tk_s,
                seg.max_tv_e, seg.max_tk_e, seg.mort_min_tick, seg.mort_max_tick
            };
            emit_bytes(s_bounds, sizeof(s_bounds));
            emit_u32(static_cast<uint32_t>(seg.v_bounds.size()));
            emit_bytes(seg.v_bounds.data(), seg.v_bounds.size() * sizeof(uint32_t));
            emit_bytes(seg.gk_counts, sizeof(seg.gk_counts));
            emit_bytes(seg.k_max_per_col, sizeof(seg.k_max_per_col));
            emit_bytes(seg.k_min_per_col, sizeof(seg.k_min_per_col));

            uint64_t nn = seg.num_nodes;
            emit_u64(nn);
            emit_bytes(seg.ext_ids.data(), nn * sizeof(uint64_t));
            emit_bytes(seg.node_gv.data(), nn * sizeof(uint8_t));
            emit_bytes(seg.node_gk.data(), nn * sizeof(uint8_t));
            emit_bytes(seg.node_cid.data(), nn * sizeof(uint8_t));
            emit_bytes(seg.node_cell_slot.data(), nn * sizeof(uint32_t));
            emit_bytes(seg.node_level.data(), nn * sizeof(uint8_t));
            emit_bytes(seg.g_tv_s.data, nn * sizeof(uint32_t));
            emit_bytes(seg.g_tv_e.data, nn * sizeof(uint32_t));
            emit_bytes(seg.g_tk_s.data, nn * sizeof(uint32_t));
            emit_bytes(seg.g_tk_e.data, nn * sizeof(uint32_t));
            emit_bytes(seg.fp32_ptr(0, cfg_.dim), nn * cfg_.dim * sizeof(float));

            // Flatten adjacency lists to CSR for sequential disk write
            auto emit_csr = [&](const std::vector<std::vector<uint32_t>>& adj) {
                std::vector<uint32_t> offs(nn + 1, 0);
                std::size_t tot_e = 0;
                for (std::size_t i = 0; i < nn; ++i) {
                    offs[i] = static_cast<uint32_t>(tot_e);
                    tot_e += adj[i].size();
                }
                offs[nn] = static_cast<uint32_t>(tot_e);
                std::vector<uint32_t> flat;
                flat.reserve(tot_e);
                for (std::size_t i = 0; i < nn; ++i) {
                    flat.insert(flat.end(), adj[i].begin(), adj[i].end());
                }
                emit_u64(tot_e);
                emit_bytes(offs.data(), (nn + 1) * sizeof(uint32_t));
                emit_bytes(flat.data(), tot_e * sizeof(uint32_t));
            };
            emit_csr(seg.adj_pre);
            emit_csr(seg.adj_fwd);

            emit_u32(static_cast<uint32_t>(seg.upper_hubs.size()));
            emit_bytes(seg.upper_hubs.data(), seg.upper_hubs.size() * sizeof(uint32_t));
            for (uint32_t uh : seg.upper_hubs) {
                uint32_t udeg = (!seg.adj_upper[uh].empty()) ? static_cast<uint32_t>(seg.adj_upper[uh][0].size()) : 0u;
                emit_u32(udeg);
                if (udeg > 0) emit_bytes(seg.adj_upper[uh][0].data(), udeg * sizeof(uint32_t));
            }

            // Emit cells
            for (const auto& c : seg.cells) {
                uint32_t c_hdr[6] = {c.min_tv_s, c.max_tv_s, c.min_tk_s, c.max_tk_s, c.max_tv_e, c.max_tk_e};
                emit_bytes(c_hdr, sizeof(c_hdr));
                emit_bytes(c.tv_end_hist, sizeof(c.tv_end_hist));
                emit_bytes(c.tk_end_hist, sizeof(c.tk_end_hist));
                uint32_t csz = static_cast<uint32_t>(c.local_ids.size());
                uint32_t pad_sz = static_cast<uint32_t>(c.tv_s.size);
                emit_u32(csz);
                emit_u32(pad_sz);
                if (pad_sz > 0) {
                    emit_bytes(c.tv_s.data, pad_sz * sizeof(uint32_t));
                    emit_bytes(c.tv_e.data, pad_sz * sizeof(uint32_t));
                    emit_bytes(c.tk_s.data, pad_sz * sizeof(uint32_t));
                    emit_bytes(c.tk_e.data, pad_sz * sizeof(uint32_t));
                }
                if (csz > 0) {
                    emit_bytes(c.local_ids.data(), csz * sizeof(uint32_t));
                    emit_bytes(c.sq8_codes.data, static_cast<std::size_t>(csz) * padded_dim_);
                    emit_bytes(c.sq8_meta.data(), csz * sizeof(simd::SQ8VectorMeta));
                }
                uint32_t esz = static_cast<uint32_t>(c.entry_nodes.size());
                emit_u32(esz);
                if (esz > 0) emit_bytes(c.entry_nodes.data(), esz * sizeof(uint32_t));
            }
        }
        flush_buf();
        if (fsync_disk) ::fsync(fd);
        ::close(fd);
        if (::rename(tmp_path.c_str(), path.c_str()) != 0) {
            ::unlink(tmp_path.c_str());
            throw std::runtime_error("Atomic rename of snapshot failed: " + path);
        }
        if (fsync_disk) {
            auto slash = path.find_last_of('/');
            std::string dir_path = (slash == std::string::npos) ? "." : (slash == 0 ? "/" : path.substr(0, slash));
            int dfd = ::open(dir_path.c_str(), O_RDONLY);
            if (dfd >= 0) {
                ::fsync(dfd);
                ::close(dfd);
            }
        }
        checkpoint_lsn_ = hdr.checkpoint_lsn;

        auto t1 = std::chrono::steady_clock::now();
        PersistenceStats st;
        st.bytes_transferred = total_written;
        st.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        st.throughput_mb_s = (st.elapsed_ms > 0.0)
            ? (static_cast<double>(total_written) / (1024.0 * 1024.0)) / (st.elapsed_ms / 1000.0)
            : 0.0;
        st.records_processed = total_nodes_;
        return st;
    }

    PersistenceStats load_snapshot(const std::string& path, bool populate_pages = true) {
        auto t0 = std::chrono::steady_clock::now();
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Failed to open snapshot for reading: " + path);
        struct stat sb{};
        if (::fstat(fd, &sb) != 0 || sb.st_size < static_cast<off_t>(sizeof(SnapshotFileHeader))) {
            ::close(fd);
            throw std::runtime_error("Invalid snapshot file size: " + path);
        }
        std::size_t file_sz = static_cast<std::size_t>(sb.st_size);
        int flags = MAP_PRIVATE;
        #ifdef MAP_POPULATE
        if (populate_pages) flags |= MAP_POPULATE;
        #endif
        void* mapped = ::mmap(nullptr, file_sz, PROT_READ, flags, fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("mmap failed on snapshot: " + path);
        }
        #ifdef MADV_SEQUENTIAL
        ::madvise(mapped, file_sz, MADV_SEQUENTIAL);
        #endif

        const uint8_t* ptr = static_cast<const uint8_t*>(mapped);
        const uint8_t* end = ptr + file_sz;
        auto read_bytes = [&](void* dst, std::size_t len) {
            if (len == 0) return;
            if (ptr + len > end) throw std::runtime_error("Corrupt truncated snapshot");
            std::memcpy(dst, ptr, len);
            ptr += len;
        };
        auto read_u64 = [&]() { uint64_t v = 0; read_bytes(&v, sizeof(v)); return v; };
        auto read_u32 = [&]() { uint32_t v = 0; read_bytes(&v, sizeof(v)); return v; };

        SnapshotFileHeader hdr{};
        read_bytes(&hdr, sizeof(hdr));
        if (hdr.magic != 0x5049544D534E5033ULL) {
            ::munmap(mapped, file_sz);
            ::close(fd);
            throw std::runtime_error("Snapshot magic mismatch");
        }
        checkpoint_lsn_ = hdr.checkpoint_lsn;
        if (checkpoint_lsn_ > wal_lsn_) wal_lsn_ = checkpoint_lsn_;

        segments_.clear();
        ext_to_loc_.clear();
        ext_to_loc_.reserve(hdr.total_nodes);
        total_nodes_ = 0;

        for (uint32_t s = 0; s < hdr.num_segments; ++s) {
            uint32_t sid = read_u32();
            uint64_t cap = read_u64();
            uint64_t ep_cap = read_u64();
            auto seg_ptr = std::make_unique<MacroSegment>(sid, cfg_.grid_bins, cap);
            MacroSegment& seg = *seg_ptr;
            seg.epoch_row_cap = ep_cap;

            uint8_t fl[3];
            read_bytes(fl, 3);
            seg.sealed = (fl[0] != 0);
            seg.bounds_ready = (fl[1] != 0);
            seg.active_gk = fl[2];

            uint32_t s_bounds[8];
            read_bytes(s_bounds, sizeof(s_bounds));
            seg.min_tv_s = s_bounds[0]; seg.max_tv_s = s_bounds[1];
            seg.min_tk_s = s_bounds[2]; seg.max_tk_s = s_bounds[3];
            seg.max_tv_e = s_bounds[4]; seg.max_tk_e = s_bounds[5];
            seg.mort_min_tick = s_bounds[6]; seg.mort_max_tick = s_bounds[7];

            uint32_t vb_sz = read_u32();
            seg.v_bounds.resize(vb_sz);
            read_bytes(seg.v_bounds.data(), vb_sz * sizeof(uint32_t));
            read_bytes(seg.gk_counts, sizeof(seg.gk_counts));
            read_bytes(seg.k_max_per_col, sizeof(seg.k_max_per_col));
            read_bytes(seg.k_min_per_col, sizeof(seg.k_min_per_col));

            uint64_t nn = read_u64();
            seg.num_nodes = static_cast<std::size_t>(nn);
            seg.linked_count.store(seg.num_nodes, std::memory_order_relaxed);
            seg.ext_ids.resize(nn);
            seg.node_gv.resize(nn);
            seg.node_gk.resize(nn);
            seg.node_cid.resize(nn);
            seg.node_cell_slot.resize(nn);
            seg.node_level.resize(nn);
            seg.g_tv_s.resize(nn);
            seg.g_tv_e.resize(nn);
            seg.g_tk_s.resize(nn);
            seg.g_tk_e.resize(nn);
            seg.fp32_vecs.resize(nn * cfg_.dim);

            read_bytes(seg.ext_ids.data(), nn * sizeof(uint64_t));
            read_bytes(seg.node_gv.data(), nn * sizeof(uint8_t));
            read_bytes(seg.node_gk.data(), nn * sizeof(uint8_t));
            read_bytes(seg.node_cid.data(), nn * sizeof(uint8_t));
            read_bytes(seg.node_cell_slot.data(), nn * sizeof(uint32_t));
            read_bytes(seg.node_level.data(), nn * sizeof(uint8_t));
            read_bytes(seg.g_tv_s.data, nn * sizeof(uint32_t));
            read_bytes(seg.g_tv_e.data, nn * sizeof(uint32_t));
            read_bytes(seg.g_tk_s.data, nn * sizeof(uint32_t));
            read_bytes(seg.g_tk_e.data, nn * sizeof(uint32_t));
            read_bytes(seg.fp32_vecs.data, nn * cfg_.dim * sizeof(float));

            std::vector<SpinLock> new_locks(nn);
            seg.node_locks.swap(new_locks);

            uint64_t seg_prefix = static_cast<uint64_t>(sid) << 32;
            for (std::size_t i = 0; i < nn; ++i) {
                ext_to_loc_[seg.ext_ids[i]] = seg_prefix | static_cast<uint64_t>(i);
                uint8_t gv_i = seg.node_gv[i];
                if (gv_i < MAX_BINS) seg.gv_counts[gv_i]++;
                uint32_t vs_i = seg.g_tv_s[i];
                uint32_t ks_i = seg.g_tk_s[i];
                if (vs_i < hist_min_tv_) hist_min_tv_ = vs_i;
                if (vs_i > hist_max_tv_) hist_max_tv_ = vs_i;
                if (ks_i < hist_min_tk_) hist_min_tk_ = ks_i;
                if (ks_i > hist_max_tk_) hist_max_tk_ = ks_i;
                record_causal_lag(vs_i, ks_i);
            }

            auto load_csr = [&](std::vector<std::vector<uint32_t>>& adj) {
                uint64_t tot_e = read_u64();
                std::vector<uint32_t> offs(nn + 1);
                read_bytes(offs.data(), (nn + 1) * sizeof(uint32_t));
                const uint32_t* flat_ptr = reinterpret_cast<const uint32_t*>(ptr);
                if (ptr + tot_e * sizeof(uint32_t) > end) throw std::runtime_error("Corrupt CSR snapshot");
                ptr += tot_e * sizeof(uint32_t);
                adj.resize(nn);
                for (std::size_t i = 0; i < nn; ++i) {
                    uint32_t b = offs[i], e = offs[i + 1];
                    adj[i].assign(flat_ptr + b, flat_ptr + e);
                }
            };
            load_csr(seg.adj_pre);
            load_csr(seg.adj_fwd);

            seg.adj_upper.resize(nn);
            for (std::size_t i = 0; i < nn; ++i) {
                if (seg.node_level[i] > 0) seg.adj_upper[i].resize(seg.node_level[i]);
            }
            uint32_t uh_sz = read_u32();
            seg.upper_hubs.resize(uh_sz);
            read_bytes(seg.upper_hubs.data(), uh_sz * sizeof(uint32_t));
            for (uint32_t uh : seg.upper_hubs) {
                uint32_t udeg = read_u32();
                if (udeg > 0) {
                    if (seg.adj_upper[uh].empty()) seg.adj_upper[uh].resize(1);
                    seg.adj_upper[uh][0].resize(udeg);
                    read_bytes(seg.adj_upper[uh][0].data(), udeg * sizeof(uint32_t));
                }
            }

            for (auto& c : seg.cells) {
                uint32_t c_hdr[6];
                read_bytes(c_hdr, sizeof(c_hdr));
                c.min_tv_s = c_hdr[0]; c.max_tv_s = c_hdr[1];
                c.min_tk_s = c_hdr[2]; c.max_tk_s = c_hdr[3];
                c.max_tv_e = c_hdr[4]; c.max_tk_e = c_hdr[5];
                read_bytes(c.tv_end_hist, sizeof(c.tv_end_hist));
                read_bytes(c.tk_end_hist, sizeof(c.tk_end_hist));
                uint32_t csz = read_u32();
                uint32_t pad_sz = read_u32();
                c.tv_s.resize(pad_sz);
                c.tv_e.resize(pad_sz);
                c.tk_s.resize(pad_sz);
                c.tk_e.resize(pad_sz);
                if (pad_sz > 0) {
                    read_bytes(c.tv_s.data, pad_sz * sizeof(uint32_t));
                    read_bytes(c.tv_e.data, pad_sz * sizeof(uint32_t));
                    read_bytes(c.tk_s.data, pad_sz * sizeof(uint32_t));
                    read_bytes(c.tk_e.data, pad_sz * sizeof(uint32_t));
                }
                c.local_ids.resize(csz);
                c.sq8_codes.resize(static_cast<std::size_t>(csz) * padded_dim_);
                c.sq8_meta.resize(csz);
                if (csz > 0) {
                    read_bytes(c.local_ids.data(), csz * sizeof(uint32_t));
                    read_bytes(c.sq8_codes.data, static_cast<std::size_t>(csz) * padded_dim_);
                    read_bytes(c.sq8_meta.data(), csz * sizeof(simd::SQ8VectorMeta));
                }
                uint32_t esz = read_u32();
                c.entry_nodes.resize(esz);
                if (esz > 0) read_bytes(c.entry_nodes.data(), esz * sizeof(uint32_t));

                std::size_t seed_cnt = std::min<std::size_t>(c.local_ids.size(), 16);
                auto& cs = seg.cell_seeds[c.gk * cfg_.grid_bins + c.gv];
                cs.assign(c.local_ids.begin(), c.local_ids.begin() + seed_cnt);
            }

            total_nodes_ += seg.num_nodes;
            segments_.push_back(std::move(seg_ptr));
        }

        ::munmap(mapped, file_sz);
        ::close(fd);

        auto t1 = std::chrono::steady_clock::now();
        PersistenceStats st;
        st.bytes_transferred = file_sz;
        st.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        st.throughput_mb_s = (st.elapsed_ms > 0.0)
            ? (static_cast<double>(file_sz) / (1024.0 * 1024.0)) / (st.elapsed_ms / 1000.0)
            : 0.0;
        st.records_processed = total_nodes_;
        return st;
    }

    void open_wal(const std::string& wal_path, bool sync_each_commit = false) {
        close_wal();
        // If opening an existing WAL stream, advance wal_lsn_ past any valid persisted records
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
                while (cur + sizeof(WALRecordHeader) <= off) {
                    WALRecordHeader hdr{};
                    std::memcpy(&hdr, buf.data() + cur, sizeof(hdr));
                    if (hdr.magic != 0x5057414Cu || cur + sizeof(WALRecordHeader) + hdr.payload_bytes > off) break;
                    if (hdr.lsn > wal_lsn_) wal_lsn_ = hdr.lsn;
                    cur += sizeof(WALRecordHeader) + hdr.payload_bytes;
                }
            }
            ::close(rfd);
        }
        wal_fd_ = ::open(wal_path.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
        if (wal_fd_ < 0) throw std::runtime_error("Failed to open WAL file: " + wal_path);
        wal_sync_each_ = sync_each_commit;
        wal_tx_depth_ = 0;
    }

    void begin_wal_tx() {
        if (wal_tx_depth_++ == 0 && wal_fd_ >= 0) {
            append_wal_record(10, 0, 0, 0, 0, 0, nullptr, 0, false);
        }
    }

    void commit_wal_tx() {
        if (wal_tx_depth_ == 0) return;
        if (--wal_tx_depth_ == 0 && wal_fd_ >= 0) {
            append_wal_record(11, 0, 0, 0, 0, 0, nullptr, 0, wal_sync_each_);
        }
    }

    void sync_wal() {
        if (wal_fd_ >= 0) ::fdatasync(wal_fd_);
    }

    void close_wal() {
        if (wal_fd_ >= 0) {
            ::fdatasync(wal_fd_);
            ::close(wal_fd_);
            wal_fd_ = -1;
        }
        wal_tx_depth_ = 0;
    }

    PersistenceStats replay_wal(const std::string& wal_path) {
        auto t0 = std::chrono::steady_clock::now();
        int fd = ::open(wal_path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Failed to open WAL for replay: " + wal_path);
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

        auto apply_one = [&](const WALRecordHeader& hdr, const uint8_t* payload) -> bool {
            if (hdr.lsn <= checkpoint_lsn_) return false;
            if (hdr.op == 1 && hdr.payload_bytes == cfg_.dim * sizeof(float)) {
                insert_single_internal(
                    hdr.ext_id, reinterpret_cast<const float*>(payload),
                    hdr.tv_s, hdr.tv_e, hdr.tk_s, hdr.tk_e, false);
                return true;
            } else if (hdr.op == 2) {
                update_valid_end_internal(hdr.ext_id, hdr.tv_e, false);
                return true;
            } else if (hdr.op == 3) {
                update_tx_end_internal(hdr.ext_id, hdr.tk_e, false);
                return true;
            }
            return false;
        };

        std::size_t cur = 0;
        std::size_t last_committed_end = 0;
        uint64_t replayed = 0;
        bool in_tx = false;
        std::vector<std::pair<WALRecordHeader, const uint8_t*>> tx_buf;

        while (cur + sizeof(WALRecordHeader) <= off) {
            WALRecordHeader hdr{};
            std::memcpy(&hdr, buf.data() + cur, sizeof(hdr));
            if (hdr.magic != 0x5057414Cu) break;
            if (cur + sizeof(WALRecordHeader) + hdr.payload_bytes > off) break; // torn tail tolerance!

            uint32_t saved_crc = hdr.crc32;
            hdr.crc32 = 0;
            uint32_t calc_crc = crc32_ieee(&hdr, sizeof(hdr), 0u);
            const uint8_t* payload = buf.data() + cur + sizeof(hdr);
            if (hdr.payload_bytes > 0) calc_crc = crc32_ieee(payload, hdr.payload_bytes, calc_crc);
            if (calc_crc != saved_crc) break; // stop cleanly at corrupt torn write

            std::size_t next_cur = cur + sizeof(WALRecordHeader) + hdr.payload_bytes;
            if (hdr.lsn > wal_lsn_) wal_lsn_ = hdr.lsn;

            if (hdr.op == 10) {
                in_tx = true;
                tx_buf.clear();
            } else if (hdr.op == 11) {
                if (in_tx) {
                    for (const auto& [thdr, tpay] : tx_buf) {
                        if (apply_one(thdr, tpay)) replayed++;
                    }
                    tx_buf.clear();
                    in_tx = false;
                }
                last_committed_end = next_cur;
            } else {
                if (in_tx) {
                    tx_buf.emplace_back(hdr, payload);
                } else {
                    if (apply_one(hdr, payload)) replayed++;
                    last_committed_end = next_cur;
                }
            }
            cur = next_cur;
        }

        // Truncate any torn record tail or uncommitted multi-op transaction suffix so subsequent appends align cleanly!
        if (last_committed_end < file_sz) {
            int t_rc = ::truncate(wal_path.c_str(), static_cast<off_t>(last_committed_end));
            (void)t_rc;
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

    // Diagnostic cell occupancy inspector on active tail segment
    std::vector<uint64_t> active_segment_cell_counts() const {
        std::vector<uint64_t> cnts(cfg_.grid_bins * cfg_.grid_bins, 0);
        if (segments_.empty()) return cnts;
        const MacroSegment& seg = *segments_.back();
        for (std::size_t i = 0; i < seg.cells.size(); ++i) cnts[i] = seg.cells[i].size();
        return cnts;
    }

    // Corpus-wide aggregated cell occupancy across all sealed + active MacroSegments
    std::vector<uint64_t> corpus_cell_counts() const {
        std::vector<uint64_t> cnts(cfg_.grid_bins * cfg_.grid_bins, 0);
        for (const auto& sp : segments_) {
            const MacroSegment& seg = *sp;
            for (std::size_t i = 0; i < seg.cells.size(); ++i) cnts[i] += seg.cells[i].size();
        }
        return cnts;
    }

    // Aggregated cell occupancy across all sealed MacroSegments (or active segment if only 1 segment exists)
    std::vector<uint64_t> sealed_segments_cell_counts() const {
        std::vector<uint64_t> cnts(cfg_.grid_bins * cfg_.grid_bins, 0);
        if (segments_.empty()) return cnts;
        bool any_sealed = false;
        for (const auto& sp : segments_) {
            const MacroSegment& seg = *sp;
            if (!seg.sealed) continue;
            any_sealed = true;
            for (std::size_t i = 0; i < seg.cells.size(); ++i) cnts[i] += seg.cells[i].size();
        }
        if (!any_sealed) {
            const MacroSegment& seg = *segments_.front();
            for (std::size_t i = 0; i < seg.cells.size(); ++i) cnts[i] += seg.cells[i].size();
        }
        return cnts;
    }

private:
    static constexpr std::size_t LAG_RESERVOIR_CAP = 4096;

    BiTHNSWConfig cfg_;
    std::size_t padded_dim_ = 0;
    double level_mult_ = 0.36;

    bool has_manual_bounds_ = false;
    std::vector<uint32_t> manual_v_bounds_;
    std::vector<uint32_t> manual_k_bounds_;
    uint32_t manual_min_tick_ = 0;
    uint32_t manual_max_tick_ = 1000000;

    std::size_t total_nodes_ = 0;
    std::vector<std::unique_ptr<MacroSegment>> segments_;
    // Maps 64-bit ext_id -> ((uint64_t(seg_id) << 32) | uint64_t(local_id))
    std::unordered_map<uint64_t, uint64_t> ext_to_loc_;

    // Causal summary & bounded Vitter lag reservoir of prior history (strictly zero future knowledge)
    uint32_t hist_min_tv_ = INF_TICK;
    uint32_t hist_max_tv_ = 0;
    uint32_t hist_min_tk_ = INF_TICK;
    uint32_t hist_max_tk_ = 0;
    uint64_t hist_lag_sum_ = 0;
    uint64_t hist_lag_cnt_ = 0;
    std::vector<uint32_t> causal_lag_reservoir_;
    uint64_t causal_lag_seen_ = 0;
    std::mt19937 lag_rng_{1337u};

    inline void record_causal_lag(uint32_t vs, uint32_t ks) {
        uint32_t lag = (ks >= vs) ? (ks - vs) : 0u;
        if (ks >= vs) {
            hist_lag_sum_ += static_cast<uint64_t>(lag);
            hist_lag_cnt_++;
        }
        causal_lag_seen_++;
        if (causal_lag_reservoir_.size() < LAG_RESERVOIR_CAP) {
            causal_lag_reservoir_.push_back(lag);
        } else {
            std::uniform_int_distribution<uint64_t> dist(0, causal_lag_seen_ - 1);
            uint64_t pos = dist(lag_rng_);
            if (pos < LAG_RESERVOIR_CAP) {
                causal_lag_reservoir_[static_cast<std::size_t>(pos)] = lag;
            }
        }
    }

    int wal_fd_ = -1;
    bool wal_sync_each_ = false;
    uint32_t wal_tx_depth_ = 0;
    uint64_t wal_lsn_ = 0;
    mutable uint64_t checkpoint_lsn_ = 0;

    inline float node_sq8_dist(const MacroSegment& seg, uint32_t lid, const simd::SQ8Query& sq_q) const noexcept {
        const auto& cell = seg.cells[seg.node_cid[lid]];
        uint32_t slot = seg.node_cell_slot[lid];
        return simd::sq8_approx_l2_sqr(
            sq_q,
            cell.sq8_codes.data + static_cast<std::size_t>(slot) * padded_dim_,
            cell.sq8_meta[slot]);
    }

    inline uint8_t bin_v_seg(const MacroSegment& seg, uint32_t tv) const noexcept {
        auto it = std::upper_bound(seg.v_bounds.begin(), seg.v_bounds.end(), tv);
        return static_cast<uint8_t>(it - seg.v_bounds.begin());
    }

    inline uint8_t bin_k_seg(const MacroSegment& seg, uint32_t tk) const noexcept {
        // Transaction columns are populated monotonically; find highest column gk with k_min_per_col[gk] <= tk
        uint8_t max_col = std::min<uint8_t>(seg.active_gk, static_cast<uint8_t>(cfg_.grid_bins - 1));
        uint8_t best = 0;
        for (uint8_t gk = 0; gk <= max_col; ++gk) {
            if (seg.gk_counts[gk] > 0 && seg.k_min_per_col[gk] <= tk) {
                best = gk;
            }
        }
        return best;
    }

    inline uint32_t mort_bin_seg(const MacroSegment& seg, uint32_t end_tick) const noexcept {
        if (end_tick == INF_TICK) return MORT_BINS - 1;
        if (end_tick <= seg.mort_min_tick) return 0;
        if (end_tick >= seg.mort_max_tick) return MORT_BINS - 2;
        uint64_t span = static_cast<uint64_t>(seg.mort_max_tick - seg.mort_min_tick) + 1ULL;
        uint64_t b = (static_cast<uint64_t>(end_tick - seg.mort_min_tick) * (MORT_BINS - 1)) / span;
        return static_cast<uint32_t>(std::min<uint64_t>(MORT_BINS - 2, b));
    }

    inline uint32_t mort_bin_query_seg(const MacroSegment& seg, uint32_t q_tick) const noexcept {
        if (q_tick == INF_TICK) return MORT_BINS - 1;
        if (q_tick < seg.mort_min_tick) return 0;
        if (q_tick >= seg.mort_max_tick) return MORT_BINS - 2;
        uint64_t span = static_cast<uint64_t>(seg.mort_max_tick - seg.mort_min_tick) + 1ULL;
        uint64_t b = (static_cast<uint64_t>(q_tick + 1u - seg.mort_min_tick) * (MORT_BINS - 1)) / span;
        return static_cast<uint32_t>(std::min<uint64_t>(MORT_BINS - 2, b));
    }

    void seal_segment(MacroSegment& seg) noexcept {
        seg.sealed = true;
    }

    uint64_t next_segment_capacity() const noexcept {
        if (!cfg_.streaming_mode) return MAX_SEG_LOCAL_NODES;
        uint64_t cap_ceiling = std::min<uint64_t>(MAX_SEG_LOCAL_NODES, std::max<uint64_t>(4096ULL, cfg_.max_segment_cap));
        if (segments_.empty()) return std::min<uint64_t>(cap_ceiling, std::max<uint64_t>(4096ULL, cfg_.initial_segment_cap));
        uint64_t prev = segments_.back()->capacity;
        return std::min<uint64_t>(cap_ceiling, prev * 2ULL);
    }

    // Calibrate a newly opened MacroSegment using ONLY prior causal history (or Segment-0 bootstrap prefix <= 2,048 items)
    // via empirical causal lag-convolution quantiles: \hat{t}_v(j, q) = t_k^{(0)} + u_q * \widehat{\Delta T}_k - \Delta_j
    void calibrate_segment_causal(
        MacroSegment& seg,
        const uint32_t* boot_tv_s,
        const uint32_t* boot_tk_s,
        const std::size_t* boot_order,
        std::size_t avail_n) {
        if (seg.bounds_ready) return;

        if (!cfg_.streaming_mode && has_manual_bounds_) {
            seg.v_bounds = manual_v_bounds_;
            seg.mort_min_tick = manual_min_tick_;
            seg.mort_max_tick = manual_max_tick_;
            seg.bounds_ready = true;
            return;
        }

        seg.v_bounds.clear();
        const uint32_t B = cfg_.grid_bins;

        if (seg.seg_id == 0) {
            // Genesis Segment 0: inspect at most the initial bootstrap window (<= 2,048 records)
            std::size_t w_boot = std::min<std::size_t>(avail_n, 2048);
            if (w_boot == 0) {
                uint32_t base_t = 1514764800u; // 2018-01-01 fallback for 1-by-1 cold insert
                uint32_t step = 86400u * 30u;
                for (uint32_t b = 1; b < B; ++b) seg.v_bounds.push_back(base_t + b * step);
                seg.mort_min_tick = base_t;
                seg.mort_max_tick = base_t + B * step * 2u;
                seg.bounds_ready = true;
                return;
            }
            std::vector<uint32_t> vs(w_boot), ks(w_boot), boot_lags(w_boot);
            for (std::size_t i = 0; i < w_boot; ++i) {
                std::size_t idx = boot_order ? boot_order[i] : i;
                vs[i] = boot_tv_s[idx];
                ks[i] = boot_tk_s[idx];
                boot_lags[i] = (ks[i] >= vs[i]) ? (ks[i] - vs[i]) : 0u;
            }
            uint32_t v_lo = *std::min_element(vs.begin(), vs.end());
            auto [it_kmin, it_kmax] = std::minmax_element(ks.begin(), ks.end());
            uint32_t tk0 = *it_kmin;
            uint32_t tkw = *it_kmax;
            double dt_per_rec = (w_boot > 1 && tkw > tk0)
                ? static_cast<double>(tkw - tk0) / static_cast<double>(w_boot)
                : 300.0;
            double proj_span = std::max<double>(86400.0 * 30.0, dt_per_rec * static_cast<double>(seg.capacity));

            std::vector<uint32_t> conv_samples;
            conv_samples.reserve(w_boot * 16);
            for (uint32_t q = 0; q < 16; ++q) {
                double u_q = (static_cast<double>(q) + 0.5) / 16.0;
                int64_t tk_q = static_cast<int64_t>(tk0) + static_cast<int64_t>(u_q * proj_span);
                for (std::size_t i = 0; i < w_boot; ++i) {
                    int64_t synth = tk_q - static_cast<int64_t>(boot_lags[i]);
                    int64_t clamped = std::clamp<int64_t>(
                        synth, static_cast<int64_t>(v_lo), static_cast<int64_t>(INF_TICK - 2u));
                    conv_samples.push_back(static_cast<uint32_t>(clamped));
                }
            }
            std::sort(conv_samples.begin(), conv_samples.end());
            for (uint32_t b = 1; b < B; ++b) {
                std::size_t q_idx = (static_cast<std::size_t>(b) * conv_samples.size()) / B;
                uint32_t tick = conv_samples[std::min(q_idx, conv_samples.size() - 1)];
                if (!seg.v_bounds.empty() && tick <= seg.v_bounds.back()) {
                    tick = std::min<uint32_t>(INF_TICK - 2u, seg.v_bounds.back() + 1u);
                }
                seg.v_bounds.push_back(tick);
            }
            uint64_t v_hi_64 = static_cast<uint64_t>(tk0) + static_cast<uint64_t>(proj_span);
            uint32_t v_hi = (v_hi_64 >= INF_TICK - 1000u) ? (INF_TICK - 1000u) : static_cast<uint32_t>(v_hi_64);
            seg.mort_min_tick = std::min(v_lo, tk0);
            seg.mort_max_tick = std::min<uint32_t>(INF_TICK - 2u, v_hi + 86400u * 180u);
        } else {
            // Rollover Segment m >= 1: calibrated strictly from prior causal segment & 4,096-lag reservoir!
            const MacroSegment& prev = *segments_[seg.seg_id - 1];
            uint64_t prev_tk_span = (prev.max_tk_s > prev.min_tk_s)
                ? static_cast<uint64_t>(prev.max_tk_s - prev.min_tk_s)
                : 86400ULL * 30ULL;
            double dt_per_rec = static_cast<double>(prev_tk_span)
                / static_cast<double>(std::max<std::size_t>(1, prev.num_nodes));
            double proj_tk_span = std::max<double>(86400.0 * 30.0, dt_per_rec * static_cast<double>(seg.capacity));

            uint32_t tk0 = prev.max_tk_s;
            uint32_t v_hist_min = std::min(hist_min_tv_, prev.min_tv_s);
            uint32_t fallback_lag = (hist_lag_cnt_ > 0)
                ? static_cast<uint32_t>(std::min<uint64_t>(INF_TICK / 2u, hist_lag_sum_ / hist_lag_cnt_))
                : (86400u * 30u);

            const std::size_t n_lags = causal_lag_reservoir_.size();
            std::vector<uint32_t> conv_samples;
            conv_samples.reserve(std::max<std::size_t>(1, n_lags) * 16);
            for (uint32_t q = 0; q < 16; ++q) {
                double u_q = (static_cast<double>(q) + 0.5) / 16.0;
                int64_t tk_q = static_cast<int64_t>(tk0) + static_cast<int64_t>(u_q * proj_tk_span);
                if (n_lags > 0) {
                    for (std::size_t j = 0; j < n_lags; ++j) {
                        int64_t synth = tk_q - static_cast<int64_t>(causal_lag_reservoir_[j]);
                        int64_t clamped = std::clamp<int64_t>(
                            synth, static_cast<int64_t>(v_hist_min), static_cast<int64_t>(INF_TICK - 2u));
                        conv_samples.push_back(static_cast<uint32_t>(clamped));
                    }
                } else {
                    int64_t synth = tk_q - static_cast<int64_t>(fallback_lag);
                    int64_t clamped = std::clamp<int64_t>(
                        synth, static_cast<int64_t>(v_hist_min), static_cast<int64_t>(INF_TICK - 2u));
                    conv_samples.push_back(static_cast<uint32_t>(clamped));
                }
            }
            std::sort(conv_samples.begin(), conv_samples.end());
            for (uint32_t b = 1; b < B; ++b) {
                std::size_t q_idx = (static_cast<std::size_t>(b) * conv_samples.size()) / B;
                uint32_t tick = conv_samples[std::min(q_idx, conv_samples.size() - 1)];
                if (!seg.v_bounds.empty() && tick <= seg.v_bounds.back()) {
                    tick = std::min<uint32_t>(INF_TICK - 2u, seg.v_bounds.back() + 1u);
                }
                seg.v_bounds.push_back(tick);
            }
            uint64_t v_proj_end64 = static_cast<uint64_t>(tk0) + static_cast<uint64_t>(proj_tk_span);
            uint32_t v_proj_end = (v_proj_end64 >= INF_TICK - 1000u)
                ? (INF_TICK - 1000u)
                : static_cast<uint32_t>(v_proj_end64);
            seg.mort_min_tick = v_hist_min;
            seg.mort_max_tick = std::min<uint32_t>(INF_TICK - 2u, v_proj_end + 86400u * 180u);
        }
        if (seg.mort_max_tick <= seg.mort_min_tick) seg.mort_max_tick = seg.mort_min_tick + 86400u * 365u;
        seg.bounds_ready = true;
    }

    MacroSegment& ensure_active_segment_for_stream(
        const uint32_t* boot_tv_s,
        const uint32_t* boot_tk_s,
        const std::size_t* boot_order,
        std::size_t avail_n) {
        if (!segments_.empty()) {
            MacroSegment& back = *segments_.back();
            // All-bin skew rollover guard: monitor BOTH interior and boundary valid-time bins (0 <= gv < B).
            // If any single gv bin accumulates > 20% of segment capacity (vs 6.25% uniform expectation), seal early and recalibrate!
            uint64_t max_gv_rows = 0;
            for (uint32_t gv = 0; gv < cfg_.grid_bins; ++gv) {
                if (back.gv_counts[gv] > max_gv_rows) max_gv_rows = back.gv_counts[gv];
            }
            bool gv_skewed = cfg_.streaming_mode && (back.num_nodes >= 8192)
                && (max_gv_rows * 100ULL > back.capacity * 20ULL);

            if (!back.sealed && back.num_nodes < back.capacity && back.active_gk < cfg_.grid_bins && !gv_skewed) {
                if (!back.bounds_ready) {
                    calibrate_segment_causal(back, boot_tv_s, boot_tk_s, boot_order, avail_n);
                }
                return back;
            }
            back.sealed = true;
        }
        uint32_t new_id = static_cast<uint32_t>(segments_.size());
        uint64_t cap = next_segment_capacity();
        segments_.push_back(std::make_unique<MacroSegment>(new_id, cfg_.grid_bins, cap));
        MacroSegment& fresh = *segments_.back();
        calibrate_segment_causal(fresh, boot_tv_s, boot_tk_s, boot_order, avail_n);
        return fresh;
    }

    void append_ordered_slice_to_segment(
        MacroSegment& seg,
        const uint64_t* ext_ids,
        const float* vectors,
        const uint32_t* tv_s,
        const uint32_t* tv_e,
        const uint32_t* tk_s,
        const uint32_t* tk_e,
        const std::size_t* slice_order,
        std::size_t slice_n,
        int num_threads) {
        if (slice_n == 0) return;
        const std::size_t base_lid = seg.num_nodes;
        const std::size_t new_seg_total = base_lid + slice_n;

        seg.ext_ids.resize(new_seg_total);
        seg.node_gv.resize(new_seg_total);
        seg.node_gk.resize(new_seg_total);
        seg.node_cid.resize(new_seg_total);
        seg.node_cell_slot.resize(new_seg_total);
        seg.node_level.resize(new_seg_total);
        seg.g_tv_s.resize(new_seg_total);
        seg.g_tv_e.resize(new_seg_total);
        seg.g_tk_s.resize(new_seg_total);
        seg.g_tk_e.resize(new_seg_total);
        seg.fp32_vecs.resize(new_seg_total * cfg_.dim);

        seg.adj_pre.resize(new_seg_total);
        seg.adj_fwd.resize(new_seg_total);
        seg.adj_upper.resize(new_seg_total);
        if (seg.node_locks.size() < new_seg_total) {
            std::size_t grow_cap = std::max<std::size_t>(
                new_seg_total, seg.node_locks.empty() ? new_seg_total : seg.node_locks.size() * 2);
            std::vector<SpinLock> new_locks(grow_cap);
            seg.node_locks.swap(new_locks);
        }

        std::vector<std::vector<uint32_t>> cell_new_lids(cfg_.grid_bins * cfg_.grid_bins);
        std::mt19937 rng(42u + seg.seg_id * 1337u + static_cast<uint32_t>(base_lid));
        std::uniform_real_distribution<double> unif(1e-9, 1.0);
        const uint64_t seg_prefix = static_cast<uint64_t>(seg.seg_id) << 32;
        ext_to_loc_.reserve(total_nodes_ + slice_n);

        for (std::size_t i = 0; i < slice_n; ++i) {
            std::size_t src = slice_order[i];
            uint32_t lid = static_cast<uint32_t>(base_lid + i);
            uint32_t vs = tv_s[src], ve = tv_e[src], ks = tk_s[src], ke = tk_e[src];

            uint8_t gk = 0;
            if (!cfg_.streaming_mode && has_manual_bounds_ && !manual_k_bounds_.empty()) {
                auto it_k = std::upper_bound(manual_k_bounds_.begin(), manual_k_bounds_.end(), ks);
                gk = static_cast<uint8_t>(it_k - manual_k_bounds_.begin());
                if (gk > seg.active_gk) seg.active_gk = gk;
            } else {
                // Online transaction-epoch column progression (strictly causal row-count rollover!)
                if (static_cast<uint32_t>(seg.active_gk) + 1u < cfg_.grid_bins && seg.gk_counts[seg.active_gk] >= seg.epoch_row_cap) {
                    // Advance gk once current column meets budget AND timestamp has moved >= column max
                    if (ks >= seg.k_max_per_col[seg.active_gk]) {
                        seg.active_gk++;
                    }
                }
                gk = std::min<uint8_t>(seg.active_gk, static_cast<uint8_t>(cfg_.grid_bins - 1));
            }
            uint8_t gv = bin_v_seg(seg, vs);
            uint8_t cid = static_cast<uint8_t>(gk * cfg_.grid_bins + gv);

            seg.gk_counts[gk]++;
            if (gv < MAX_BINS) seg.gv_counts[gv]++;
            if (ks < seg.k_min_per_col[gk]) seg.k_min_per_col[gk] = ks;
            if (ks > seg.k_max_per_col[gk]) seg.k_max_per_col[gk] = ks;

            if (vs < seg.min_tv_s) seg.min_tv_s = vs;
            if (vs > seg.max_tv_s) seg.max_tv_s = vs;
            if (ks < seg.min_tk_s) seg.min_tk_s = ks;
            if (ks > seg.max_tk_s) seg.max_tk_s = ks;
            if (ve > seg.max_tv_e) seg.max_tv_e = ve;
            if (ke > seg.max_tk_e) seg.max_tk_e = ke;

            if (vs < hist_min_tv_) hist_min_tv_ = vs;
            if (vs > hist_max_tv_) hist_max_tv_ = vs;
            if (ks < hist_min_tk_) hist_min_tk_ = ks;
            if (ks > hist_max_tk_) hist_max_tk_ = ks;
            record_causal_lag(vs, ks);

            seg.ext_ids[lid] = ext_ids[src];
            ext_to_loc_[ext_ids[src]] = seg_prefix | static_cast<uint64_t>(lid);
            seg.node_gv[lid] = gv;
            seg.node_gk[lid] = gk;
            seg.node_cid[lid] = cid;
            seg.g_tv_s[lid] = vs;
            seg.g_tv_e[lid] = ve;
            seg.g_tk_s[lid] = ks;
            seg.g_tk_e[lid] = ke;

            const float* src_v = vectors + src * cfg_.dim;
            std::memcpy(seg.fp32_vecs.data + static_cast<std::size_t>(lid) * cfg_.dim, src_v, cfg_.dim * sizeof(float));

            int lvl = static_cast<int>(-std::log(unif(rng)) * level_mult_);
            if (lvl > 4) lvl = 4;
            seg.node_level[lid] = static_cast<uint8_t>(lvl);
            seg.adj_pre[lid].reserve(cfg_.m_pre * 2 + 2);
            seg.adj_fwd[lid].reserve(cfg_.m_fwd + 2);
            if (lvl > 0) {
                seg.adj_upper[lid].resize(lvl);
                for (int l = 0; l < lvl; ++l) seg.adj_upper[lid][l].reserve(cfg_.m_upper + 1);
            }
            cell_new_lids[cid].push_back(lid);
        }

        // Populate canonical cell-contiguous SoA + SQ8 buffers (single SQ8 copy!)
        for (uint32_t cid = 0; cid < seg.cells.size(); ++cid) {
            auto& cell = seg.cells[cid];
            const auto& lids = cell_new_lids[cid];
            if (lids.empty()) continue;

            std::size_t old_sz = cell.size();
            std::size_t add_sz = lids.size();
            std::size_t pad_sz = ((old_sz + add_sz + 7u) / 8u) * 8u;
            cell.tv_s.reserve(pad_sz);
            cell.tv_e.reserve(pad_sz);
            cell.tk_s.reserve(pad_sz);
            cell.tk_e.reserve(pad_sz);
            cell.sq8_codes.reserve((old_sz + add_sz) * padded_dim_ + 64);
            cell.sq8_meta.reserve(old_sz + add_sz);
            cell.local_ids.reserve(old_sz + add_sz);

            cell.tv_s.size = old_sz;
            cell.tv_e.size = old_sz;
            cell.tk_s.size = old_sz;
            cell.tk_e.size = old_sz;

            for (uint32_t lid : lids) {
                uint32_t slot = static_cast<uint32_t>(cell.local_ids.size());
                seg.node_cell_slot[lid] = slot;
                cell.local_ids.push_back(lid);

                uint32_t vs = seg.g_tv_s[lid], ve = seg.g_tv_e[lid];
                uint32_t ks = seg.g_tk_s[lid], ke = seg.g_tk_e[lid];
                cell.tv_s.push_back(vs);
                cell.tv_e.push_back(ve);
                cell.tk_s.push_back(ks);
                cell.tk_e.push_back(ke);

                if (vs < cell.min_tv_s) cell.min_tv_s = vs;
                if (vs > cell.max_tv_s) cell.max_tv_s = vs;
                if (ks < cell.min_tk_s) cell.min_tk_s = ks;
                if (ks > cell.max_tk_s) cell.max_tk_s = ks;
                if (ve > cell.max_tv_e) cell.max_tv_e = ve;
                if (ke > cell.max_tk_e) cell.max_tk_e = ke;

                cell.tv_end_hist[mort_bin_seg(seg, ve)]++;
                cell.tk_end_hist[mort_bin_seg(seg, ke)]++;

                std::size_t code_off = static_cast<std::size_t>(slot) * padded_dim_;
                cell.sq8_codes.resize(code_off + padded_dim_, 0);
                const float* nv = seg.fp32_vecs.data + static_cast<std::size_t>(lid) * cfg_.dim;
                cell.sq8_meta.push_back(simd::quantize_db_sq8(nv, cell.sq8_codes.data + code_off, cfg_.dim));

                std::size_t hub_stride = std::max<std::size_t>(1, (old_sz + add_sz) / 48);
                if (cell.entry_nodes.size() < 64 && (slot % hub_stride == 0 || seg.node_level[lid] > 1)) {
                    cell.entry_nodes.push_back(lid);
                }
            }
            while ((cell.tv_s.size & 7u) != 0) {
                cell.tv_s.push_back(INF_TICK);
                cell.tv_e.push_back(0u);
                cell.tk_s.push_back(INF_TICK);
                cell.tk_e.push_back(0u);
            }
        }

        seg.num_nodes = new_seg_total;
        total_nodes_ += slice_n;

        // Topological causal epoch-by-epoch HNSW linking within segment
        #ifdef _OPENMP
        int prev_threads = omp_get_max_threads();
        if (num_threads > 0) omp_set_num_threads(num_threads);
        #endif

        for (uint32_t gk = 0; gk < cfg_.grid_bins; ++gk) {
            for (uint32_t gv = 0; gv < cfg_.grid_bins; ++gv) {
                const auto& lids = cell_new_lids[gk * cfg_.grid_bins + gv];
                if (lids.empty()) continue;

                std::size_t boot_cnt = std::min<std::size_t>(lids.size(), 8);
                {
                    if (seg.single_vis.size() < seg.num_nodes) {
                        seg.single_vis.resize(std::max<std::size_t>(seg.num_nodes, seg.single_vis.size() * 2), 0);
                    }
                    for (std::size_t p = 0; p < boot_cnt; ++p) {
                        uint32_t lid = lids[p];
                        if (++seg.single_vis_tok == 0) {
                            std::fill(seg.single_vis.begin(), seg.single_vis.end(), 0);
                            seg.single_vis_tok = 1;
                        }
                        if (seg.linked_count.load(std::memory_order_relaxed) > 0) {
                            link_node_causal(seg, lid, static_cast<uint8_t>(gv), static_cast<uint8_t>(gk), seg.single_vis, seg.single_vis_tok);
                        }
                        seg.linked_count.fetch_add(1, std::memory_order_relaxed);
                        mark_cell_ready(seg, static_cast<uint8_t>(gv), static_cast<uint8_t>(gk), lid);
                    }
                }
                std::size_t start_pos = boot_cnt;

                #pragma omp parallel if(lids.size() > 64)
                {
                    std::vector<uint32_t> visited(seg.num_nodes, 0);
                    uint32_t vis_tok = 1;
                    #pragma omp for schedule(dynamic, 32)
                    for (std::size_t p = start_pos; p < lids.size(); ++p) {
                        uint32_t lid = lids[p];
                        if (++vis_tok == 0) {
                            std::fill(visited.begin(), visited.end(), 0);
                            vis_tok = 1;
                        }
                        link_node_causal(seg, lid, static_cast<uint8_t>(gv), static_cast<uint8_t>(gk), visited, vis_tok);
                        seg.linked_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                for (std::size_t p = boot_cnt; p < std::min<std::size_t>(lids.size(), 16); ++p) {
                    mark_cell_ready(seg, static_cast<uint8_t>(gv), static_cast<uint8_t>(gk), lids[p]);
                }

                // Link Layer-1 Causal Highway hubs
                std::vector<uint32_t> cell_l1;
                for (uint32_t lid : lids) {
                    if (seg.node_level[lid] >= 1) cell_l1.push_back(lid);
                }
                if (!cell_l1.empty()) {
                    std::size_t l1_boot = std::min<std::size_t>(cell_l1.size(), 2);
                    {
                        if (seg.single_vis.size() < seg.num_nodes) {
                            seg.single_vis.resize(std::max<std::size_t>(seg.num_nodes, seg.single_vis.size() * 2), 0);
                        }
                        for (std::size_t b = 0; b < l1_boot; ++b) {
                            if (++seg.single_vis_tok == 0) {
                                std::fill(seg.single_vis.begin(), seg.single_vis.end(), 0);
                                seg.single_vis_tok = 1;
                            }
                            if (!seg.upper_hubs.empty()) {
                                link_upper_causal(seg, cell_l1[b], static_cast<uint8_t>(gv), static_cast<uint8_t>(gk), seg.single_vis, seg.single_vis_tok);
                            }
                            seg.upper_hubs.push_back(cell_l1[b]);
                        }
                    }
                    #pragma omp parallel if(cell_l1.size() > 32)
                    {
                        std::vector<uint32_t> uvis(seg.num_nodes, 0);
                        uint32_t utok = 1;
                        #pragma omp for schedule(dynamic, 16)
                        for (std::size_t b = l1_boot; b < cell_l1.size(); ++b) {
                            if (++utok == 0) { std::fill(uvis.begin(), uvis.end(), 0); utok = 1; }
                            link_upper_causal(seg, cell_l1[b], static_cast<uint8_t>(gv), static_cast<uint8_t>(gk), uvis, utok);
                        }
                    }
                    for (std::size_t b = l1_boot; b < cell_l1.size(); ++b) {
                        seg.upper_hubs.push_back(cell_l1[b]);
                    }
                }
            }
        }

        #ifdef _OPENMP
        if (num_threads > 0) omp_set_num_threads(prev_threads);
        #endif
    }

    void insert_single_internal(
        uint64_t ext_id,
        const float* vec,
        uint32_t tv_s, uint32_t tv_e,
        uint32_t tk_s, uint32_t tk_e,
        bool write_wal) {
        if (write_wal && wal_fd_ >= 0) {
            append_wal_record(1, ext_id, tv_s, tv_e, tk_s, tk_e, vec, cfg_.dim * sizeof(float), wal_sync_each_ && wal_tx_depth_ == 0);
        }
        std::size_t zero_ord = 0;
        MacroSegment& seg = ensure_active_segment_for_stream(&tv_s, &tk_s, &zero_ord, 1);
        append_ordered_slice_to_segment(seg, &ext_id, vec, &tv_s, &tv_e, &tk_s, &tk_e, &zero_ord, 1, 1);
        if (seg.num_nodes >= seg.capacity || seg.active_gk >= cfg_.grid_bins) {
            seal_segment(seg);
        }
    }

    bool update_valid_end_internal(uint64_t ext_id, uint32_t new_tv_e, bool write_wal) {
        auto it = ext_to_loc_.find(ext_id);
        if (it == ext_to_loc_.end()) return false;
        if (write_wal && wal_fd_ >= 0) {
            append_wal_record(2, ext_id, 0, new_tv_e, 0, 0, nullptr, 0, wal_sync_each_ && wal_tx_depth_ == 0);
        }
        uint32_t sid = static_cast<uint32_t>(it->second >> 32);
        uint32_t lid = static_cast<uint32_t>(it->second & 0xFFFFFFFFu);
        MacroSegment& seg = *segments_[sid];
        uint32_t old_tv_e = seg.g_tv_e[lid];
        seg.g_tv_e[lid] = new_tv_e;
        if (new_tv_e > seg.max_tv_e) seg.max_tv_e = new_tv_e;

        auto& cell = seg.cells[seg.node_cid[lid]];
        uint32_t slot = seg.node_cell_slot[lid];
        cell.tv_e[slot] = new_tv_e;
        if (new_tv_e > cell.max_tv_e) cell.max_tv_e = new_tv_e;
        uint32_t old_b = mort_bin_seg(seg, old_tv_e);
        uint32_t new_b = mort_bin_seg(seg, new_tv_e);
        if (old_b != new_b) {
            if (cell.tv_end_hist[old_b] > 0) cell.tv_end_hist[old_b]--;
            cell.tv_end_hist[new_b]++;
        }
        return true;
    }

    bool update_tx_end_internal(uint64_t ext_id, uint32_t new_tk_e, bool write_wal) {
        auto it = ext_to_loc_.find(ext_id);
        if (it == ext_to_loc_.end()) return false;
        if (write_wal && wal_fd_ >= 0) {
            append_wal_record(3, ext_id, 0, 0, 0, new_tk_e, nullptr, 0, wal_sync_each_ && wal_tx_depth_ == 0);
        }
        uint32_t sid = static_cast<uint32_t>(it->second >> 32);
        uint32_t lid = static_cast<uint32_t>(it->second & 0xFFFFFFFFu);
        MacroSegment& seg = *segments_[sid];
        uint32_t old_tk_e = seg.g_tk_e[lid];
        seg.g_tk_e[lid] = new_tk_e;
        if (new_tk_e > seg.max_tk_e) seg.max_tk_e = new_tk_e;

        auto& cell = seg.cells[seg.node_cid[lid]];
        uint32_t slot = seg.node_cell_slot[lid];
        cell.tk_e[slot] = new_tk_e;
        if (new_tk_e > cell.max_tk_e) cell.max_tk_e = new_tk_e;
        uint32_t old_b = mort_bin_seg(seg, old_tk_e);
        uint32_t new_b = mort_bin_seg(seg, new_tk_e);
        if (old_b != new_b) {
            if (cell.tk_end_hist[old_b] > 0) cell.tk_end_hist[old_b]--;
            cell.tk_end_hist[new_b]++;
        }
        return true;
    }

    void append_wal_record(
        uint8_t op, uint64_t ext_id,
        uint32_t tv_s, uint32_t tv_e, uint32_t tk_s, uint32_t tk_e,
        const void* payload, std::size_t payload_len,
        bool sync_now) {
        WALRecordHeader hdr{};
        hdr.op = op;
        hdr.lsn = ++wal_lsn_;
        hdr.ext_id = ext_id;
        hdr.tv_s = tv_s;
        hdr.tv_e = tv_e;
        hdr.tk_s = tk_s;
        hdr.tk_e = tk_e;
        hdr.payload_bytes = static_cast<uint32_t>(payload_len);
        hdr.crc32 = 0;
        uint32_t crc = crc32_ieee(&hdr, sizeof(hdr), 0u);
        if (payload_len > 0 && payload != nullptr) {
            crc = crc32_ieee(payload, payload_len, crc);
        }
        hdr.crc32 = crc;
        ssize_t w1 = ::write(wal_fd_, &hdr, sizeof(hdr));
        if (w1 != static_cast<ssize_t>(sizeof(hdr))) {
            throw std::runtime_error("WAL header write failed");
        }
        if (payload_len > 0 && payload != nullptr) {
            const uint8_t* p = static_cast<const uint8_t*>(payload);
            std::size_t rem = payload_len;
            while (rem > 0) {
                ssize_t w2 = ::write(wal_fd_, p, rem);
                if (w2 <= 0) throw std::runtime_error("WAL payload write failed");
                p += static_cast<std::size_t>(w2);
                rem -= static_cast<std::size_t>(w2);
            }
        }
        if (sync_now) ::fdatasync(wal_fd_);
    }

    void mark_cell_ready(MacroSegment& seg, uint8_t gv, uint8_t gk, uint32_t lid) {
        seg.ready_lock.lock();
        auto& seeds = seg.cell_seeds[gk * cfg_.grid_bins + gv];
        if (seeds.size() < 16) seeds.push_back(lid);
        seg.ready_lock.unlock();
    }

    void finalize_exact_rerank_global(
        const float* q_vec,
        uint32_t k,
        std::priority_queue<std::pair<float, uint64_t>>& sq_candidates,
        QueryResult& out) const {
        using ExactPair = std::pair<float, uint64_t>;
        std::priority_queue<ExactPair> exact_top;
        while (!sq_candidates.empty()) {
            uint64_t gh = sq_candidates.top().second;
            sq_candidates.pop();
            uint32_t sid = static_cast<uint32_t>(gh >> 32);
            uint32_t lid = static_cast<uint32_t>(gh & 0xFFFFFFFFu);
            const MacroSegment& seg = *segments_[sid];
            float ed = simd::l2_sqr_fp32(
                q_vec, seg.fp32_ptr(lid, cfg_.dim), cfg_.dim);
            out.stats.distance_computations++;
            if (exact_top.size() < k) {
                exact_top.emplace(ed, seg.ext_ids[lid]);
            } else if (ed < exact_top.top().first) {
                exact_top.pop();
                exact_top.emplace(ed, seg.ext_ids[lid]);
            }
        }
        out.ids.resize(exact_top.size());
        out.distances.resize(exact_top.size());
        for (std::size_t idx = exact_top.size(); idx > 0; --idx) {
            out.ids[idx - 1] = exact_top.top().second;
            out.distances[idx - 1] = exact_top.top().first;
            exact_top.pop();
        }
    }

    void select_diverse_neighbors(
        const MacroSegment& seg,
        const std::vector<std::pair<float, uint32_t>>& sorted_cands,
        uint32_t max_m,
        std::vector<uint32_t>& out_packed) const {
        out_packed.clear();
        out_packed.reserve(max_m);
        std::vector<uint32_t> chosen_ids;
        chosen_ids.reserve(max_m);

        for (const auto& [dist_to_u, cand_id] : sorted_cands) {
            if (chosen_ids.size() >= max_m) break;
            bool good = true;
            const float* c_vec = seg.fp32_ptr(cand_id, cfg_.dim);
            for (uint32_t ex_id : chosen_ids) {
                float d_inter = simd::l2_sqr_fp32(
                    c_vec, seg.fp32_ptr(ex_id, cfg_.dim), cfg_.dim);
                if (d_inter < dist_to_u) {
                    good = false;
                    break;
                }
            }
            if (good) {
                chosen_ids.push_back(cand_id);
                out_packed.push_back(pack_neighbor(cand_id, seg.node_gv[cand_id], seg.node_gk[cand_id]));
            }
        }
        if (chosen_ids.size() < max_m) {
            for (const auto& [dist_to_u, cand_id] : sorted_cands) {
                if (chosen_ids.size() >= max_m) break;
                if (std::find(chosen_ids.begin(), chosen_ids.end(), cand_id) != chosen_ids.end()) continue;
                const float* c_vec = seg.fp32_ptr(cand_id, cfg_.dim);
                bool near_clone = false;
                for (uint32_t ex_id : chosen_ids) {
                    if (simd::l2_sqr_fp32(c_vec, seg.fp32_ptr(ex_id, cfg_.dim), cfg_.dim) < 1e-4f) {
                        near_clone = true;
                        break;
                    }
                }
                if (!near_clone) {
                    chosen_ids.push_back(cand_id);
                    out_packed.push_back(pack_neighbor(cand_id, seg.node_gv[cand_id], seg.node_gk[cand_id]));
                }
            }
        }
    }

    void add_reverse_edge_locked(
        MacroSegment& seg,
        uint32_t target,
        uint32_t src,
        uint8_t src_gv,
        uint8_t src_gk,
        bool into_prefix_list) {
        uint32_t max_deg = into_prefix_list ? (cfg_.m_pre * 2) : cfg_.m_fwd;
        seg.node_locks[target].lock();
        auto& lst = into_prefix_list ? seg.adj_pre[target] : seg.adj_fwd[target];
        uint32_t packed_src = pack_neighbor(src, src_gv, src_gk);
        for (uint32_t ex : lst) {
            if (unpack_id(ex) == src) {
                seg.node_locks[target].unlock();
                return;
            }
        }
        if (lst.size() < max_deg) {
            lst.push_back(packed_src);
        } else {
            const float* t_vec = seg.fp32_ptr(target, cfg_.dim);
            std::vector<std::pair<float, uint32_t>> cands;
            cands.reserve(lst.size() + 1);
            for (uint32_t ex : lst) {
                uint32_t eid = unpack_id(ex);
                float d = simd::l2_sqr_fp32(t_vec, seg.fp32_ptr(eid, cfg_.dim), cfg_.dim);
                cands.emplace_back(d, eid);
            }
            float d_new = simd::l2_sqr_fp32(t_vec, seg.fp32_ptr(src, cfg_.dim), cfg_.dim);
            cands.emplace_back(d_new, src);
            std::sort(cands.begin(), cands.end());
            select_diverse_neighbors(seg, cands, max_deg, lst);
        }
        seg.node_locks[target].unlock();
    }

    void link_node_causal(
        MacroSegment& seg,
        uint32_t u,
        uint8_t u_gv,
        uint8_t u_gk,
        std::vector<uint32_t>& visited,
        uint32_t vis_tok) {
        const float* u_vec = seg.fp32_ptr(u, cfg_.dim);
        simd::SQ8Query sq_u = simd::prepare_sq8_query(u_vec, cfg_.dim);

        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> top_ef;

        visited[u] = vis_tok;

        seg.ready_lock.lock();
        uint32_t seed_ids[24];
        uint32_t n_seeds = 0;
        for (uint32_t sid : seg.cell_seeds[u_gk * cfg_.grid_bins + u_gv]) {
            if (sid != u && n_seeds < 10) seed_ids[n_seeds++] = sid;
        }
        for (int gk = static_cast<int>(u_gk); gk >= 0 && n_seeds < 24; --gk) {
            for (int gv = static_cast<int>(u_gv); gv >= 0 && n_seeds < 24; --gv) {
                if (gk == u_gk && gv == u_gv) continue;
                const auto& cs = seg.cell_seeds[gk * cfg_.grid_bins + gv];
                if (!cs.empty()) seed_ids[n_seeds++] = cs[0];
            }
        }
        seg.ready_lock.unlock();

        if (n_seeds == 0) {
            mark_cell_ready(seg, u_gv, u_gk, u);
            return;
        }

        for (uint32_t s = 0; s < n_seeds; ++s) {
            uint32_t sid = seed_ids[s];
            if (visited[sid] == vis_tok) continue;
            visited[sid] = vis_tok;
            float d = node_sq8_dist(seg, sid, sq_u);
            cand_min.emplace(d, sid);
            top_ef.emplace(d, sid);
        }

        uint32_t expansions = 0;
        const uint32_t max_exp = cfg_.ef_construction * 6;
        while (!cand_min.empty() && expansions < max_exp) {
            auto [cd, curr] = cand_min.top();
            cand_min.pop();
            if (top_ef.size() >= cfg_.ef_construction && cd > top_ef.top().first) break;
            expansions++;

            uint32_t local_nbs[64];
            uint32_t n_nbs = 0;
            seg.node_locks[curr].lock();
            for (uint32_t p : seg.adj_pre[curr]) {
                if (n_nbs < 64) local_nbs[n_nbs++] = p;
            }
            for (uint32_t p : seg.adj_fwd[curr]) {
                if (unpack_gv(p) <= u_gv && unpack_gk(p) <= u_gk && n_nbs < 64) {
                    local_nbs[n_nbs++] = p;
                }
            }
            seg.node_locks[curr].unlock();

            for (uint32_t j = 0; j < n_nbs; ++j) {
                uint32_t nb = unpack_id(local_nbs[j]);
                if (visited[nb] == vis_tok) continue;
                visited[nb] = vis_tok;
                float d = node_sq8_dist(seg, nb, sq_u);
                if (top_ef.size() < cfg_.ef_construction || d < top_ef.top().first) {
                    cand_min.emplace(d, nb);
                    top_ef.emplace(d, nb);
                    if (top_ef.size() > cfg_.ef_construction) top_ef.pop();
                }
            }
        }

        std::vector<Pair> exact_cands;
        exact_cands.reserve(top_ef.size());
        while (!top_ef.empty()) {
            uint32_t cid = top_ef.top().second;
            top_ef.pop();
            float ed = simd::l2_sqr_fp32(u_vec, seg.fp32_ptr(cid, cfg_.dim), cfg_.dim);
            exact_cands.emplace_back(ed, cid);
        }
        std::sort(exact_cands.begin(), exact_cands.end());

        std::vector<uint32_t> chosen_packed;
        select_diverse_neighbors(seg, exact_cands, cfg_.m_pre, chosen_packed);

        seg.node_locks[u].lock();
        seg.adj_pre[u] = chosen_packed;
        seg.node_locks[u].unlock();

        for (uint32_t packed : chosen_packed) {
            uint32_t target = unpack_id(packed);
            bool same_cell = (seg.node_gv[target] == u_gv) && (seg.node_gk[target] == u_gk);
            add_reverse_edge_locked(seg, target, u, u_gv, u_gk, same_cell);
        }
    }

    void link_upper_causal(
        MacroSegment& seg,
        uint32_t u,
        uint8_t u_gv,
        uint8_t u_gk,
        std::vector<uint32_t>& visited,
        uint32_t vis_tok) {
        if (seg.upper_hubs.empty() || seg.adj_upper[u].empty()) return;
        const float* u_vec = seg.fp32_ptr(u, cfg_.dim);
        simd::SQ8Query sq_u = simd::prepare_sq8_query(u_vec, cfg_.dim);

        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_min;
        std::priority_queue<Pair> top_ef;
        const uint32_t ef_u = 48;
        visited[u] = vis_tok;

        std::size_t stride = std::max<std::size_t>(1, seg.upper_hubs.size() / 24);
        for (std::size_t i = 0; i < seg.upper_hubs.size(); i += stride) {
            uint32_t sid = seg.upper_hubs[i];
            if (sid == u || seg.node_gv[sid] > u_gv || seg.node_gk[sid] > u_gk || visited[sid] == vis_tok) continue;
            visited[sid] = vis_tok;
            float d = node_sq8_dist(seg, sid, sq_u);
            cand_min.emplace(d, sid);
            top_ef.emplace(d, sid);
        }

        uint32_t exp = 0;
        while (!cand_min.empty() && exp < 96) {
            auto [cd, curr] = cand_min.top();
            cand_min.pop();
            if (top_ef.size() >= ef_u && cd > top_ef.top().first) break;
            exp++;

            uint32_t lnbs[48];
            uint32_t n_lnbs = 0;
            seg.node_locks[curr].lock();
            if (!seg.adj_upper[curr].empty()) {
                for (uint32_t p : seg.adj_upper[curr][0]) {
                    if (unpack_gv(p) <= u_gv && unpack_gk(p) <= u_gk && n_lnbs < 48) lnbs[n_lnbs++] = p;
                }
            }
            seg.node_locks[curr].unlock();

            for (uint32_t j = 0; j < n_lnbs; ++j) {
                uint32_t nb = unpack_id(lnbs[j]);
                if (visited[nb] == vis_tok) continue;
                visited[nb] = vis_tok;
                float d = node_sq8_dist(seg, nb, sq_u);
                if (top_ef.size() < ef_u || d < top_ef.top().first) {
                    cand_min.emplace(d, nb);
                    top_ef.emplace(d, nb);
                    if (top_ef.size() > ef_u) top_ef.pop();
                }
            }
        }

        std::vector<Pair> cands;
        cands.reserve(top_ef.size());
        while (!top_ef.empty()) {
            uint32_t cid = top_ef.top().second;
            top_ef.pop();
            float ed = simd::l2_sqr_fp32(u_vec, seg.fp32_ptr(cid, cfg_.dim), cfg_.dim);
            cands.emplace_back(ed, cid);
        }
        std::sort(cands.begin(), cands.end());

        std::vector<uint32_t> chosen;
        select_diverse_neighbors(seg, cands, cfg_.m_upper, chosen);
        seg.node_locks[u].lock();
        seg.adj_upper[u][0] = chosen;
        seg.node_locks[u].unlock();

        uint32_t packed_u = pack_neighbor(u, u_gv, u_gk);
        for (uint32_t p : chosen) {
            uint32_t t = unpack_id(p);
            seg.node_locks[t].lock();
            if (!seg.adj_upper[t].empty() && seg.adj_upper[t][0].size() < cfg_.m_upper * 2) {
                seg.adj_upper[t][0].push_back(packed_u);
            }
            seg.node_locks[t].unlock();
        }
    }
};

} // namespace pitmem
