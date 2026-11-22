#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pit_types.hpp"
#include "simd_kernels.hpp"
#include "bit_hnsw.hpp"
#include "mvcc_provenance.hpp"
#include "rap_compact.hpp"
#include "baselines_cpp.hpp"

namespace py = pybind11;
using namespace pitmem;

PYBIND11_MODULE(pit_mem_core, m) {
    m.doc() = "PiT-Mem: Point-in-Time Bi-Temporal Storage and Indexing Engine (C++17 + AVX2 + OpenMP)";
    m.attr("INF_TICK") = py::int_(INF_TICK);

    py::class_<SearchStats>(m, "SearchStats")
        .def_readonly("latency_us", &SearchStats::latency_us)
        .def_readonly("visited_cells", &SearchStats::visited_cells)
        .def_readonly("graph_cells", &SearchStats::graph_cells)
        .def_readonly("scan_cells", &SearchStats::scan_cells)
        .def_readonly("distance_computations", &SearchStats::distance_computations)
        .def_readonly("predicate_checks", &SearchStats::predicate_checks);

    py::class_<QueryResult>(m, "QueryResult")
        .def_readonly("ids", &QueryResult::ids)
        .def_readonly("distances", &QueryResult::distances)
        .def_readonly("stats", &QueryResult::stats);

    py::class_<PersistenceStats>(m, "PersistenceStats")
        .def_readonly("bytes_transferred", &PersistenceStats::bytes_transferred)
        .def_readonly("elapsed_ms", &PersistenceStats::elapsed_ms)
        .def_readonly("throughput_mb_s", &PersistenceStats::throughput_mb_s)
        .def_readonly("records_processed", &PersistenceStats::records_processed);

    py::class_<BiTHNSWConfig>(m, "BiTHNSWConfig")
        .def(py::init<>())
        .def_readwrite("dim", &BiTHNSWConfig::dim)
        .def_readwrite("m_pre", &BiTHNSWConfig::m_pre)
        .def_readwrite("m_fwd", &BiTHNSWConfig::m_fwd)
        .def_readwrite("m_upper", &BiTHNSWConfig::m_upper)
        .def_readwrite("ef_construction", &BiTHNSWConfig::ef_construction)
        .def_readwrite("grid_bins", &BiTHNSWConfig::grid_bins)
        .def_readwrite("scan_alive_thresh", &BiTHNSWConfig::scan_alive_thresh)
        .def_readwrite("min_graph_survivor_ratio", &BiTHNSWConfig::min_graph_survivor_ratio)
        .def_readwrite("streaming_mode", &BiTHNSWConfig::streaming_mode)
        .def_readwrite("initial_segment_cap", &BiTHNSWConfig::initial_segment_cap)
        .def_readwrite("max_segment_cap", &BiTHNSWConfig::max_segment_cap)
        .def_readwrite("auto_calibrate_router", &BiTHNSWConfig::auto_calibrate_router)
        .def_readwrite("router_gamma_0", &BiTHNSWConfig::router_gamma_0)
        .def_readwrite("router_gamma_ts", &BiTHNSWConfig::router_gamma_ts)
        .def_readwrite("router_gamma_hub", &BiTHNSWConfig::router_gamma_hub)
        .def_readwrite("router_gamma_llc", &BiTHNSWConfig::router_gamma_llc)
        .def_readwrite("router_gamma_dim", &BiTHNSWConfig::router_gamma_dim)
        .def_readwrite("router_llc_knee", &BiTHNSWConfig::router_llc_knee);

    py::class_<BiTHNSWIndex>(m, "BiTHNSWIndex")
        .def(py::init<const BiTHNSWConfig&>())
        .def("calibrate_router_hardware", &BiTHNSWIndex::calibrate_router_hardware,
             py::arg("l3_cache_bytes") = 47185920ULL)
        .def("config", &BiTHNSWIndex::config)
        .def("calibrate_bounds", [](
            BiTHNSWIndex& self,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_s) {
            std::size_t n = static_cast<std::size_t>(tv_s.shape(0));
            self.calibrate_bounds(tv_s.data(), tk_s.data(), n);
        })
        .def("build_batch", [](
            BiTHNSWIndex& self,
            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_e,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_e,
            int num_threads) {
            std::size_t n = static_cast<std::size_t>(ids.shape(0));
            const uint64_t* p_ids = ids.data();
            const float* p_vecs = vecs.data();
            const uint32_t* p_tvs = tv_s.data();
            const uint32_t* p_tve = tv_e.data();
            const uint32_t* p_tks = tk_s.data();
            const uint32_t* p_tke = tk_e.data();
            py::gil_scoped_release release;
            self.build_batch(p_ids, p_vecs, p_tvs, p_tve, p_tks, p_tke, n, num_threads);
        }, py::arg("ids"), py::arg("vecs"), py::arg("tv_s"), py::arg("tv_e"),
           py::arg("tk_s"), py::arg("tk_e"), py::arg("num_threads") = 0)
        .def("insert_single", [](
            BiTHNSWIndex& self,
            uint64_t ext_id,
            py::array_t<float, py::array::c_style | py::array::forcecast> vec,
            uint32_t tv_s, uint32_t tv_e, uint32_t tk_s, uint32_t tk_e) {
            const float* pv = vec.data();
            py::gil_scoped_release release;
            self.insert_single(ext_id, pv, tv_s, tv_e, tk_s, tk_e);
        })
        .def("update_valid_end", &BiTHNSWIndex::update_valid_end)
        .def("update_tx_end", &BiTHNSWIndex::update_tx_end)
        .def("search", [](
            const BiTHNSWIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vec,
            uint32_t tv_q, uint32_t tk_q, uint32_t k, uint32_t ef_search, int force_mode) {
            const float* pq = q_vec.data();
            py::gil_scoped_release release;
            return self.search(pq, tv_q, tk_q, k, ef_search, force_mode);
        }, py::arg("q_vec"), py::arg("tv_q"), py::arg("tk_q"),
           py::arg("k") = 10, py::arg("ef_search") = 64, py::arg("force_mode") = -1)
        .def("search_batch", [](
            const BiTHNSWIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_qs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_qs,
            uint32_t k, uint32_t ef_search, int force_mode, int num_threads) {
            std::size_t nq = static_cast<std::size_t>(q_vecs.shape(0));
            std::size_t dim = static_cast<std::size_t>(q_vecs.shape(1));
            const float* pq = q_vecs.data();
            const uint32_t* ptv = tv_qs.data();
            const uint32_t* ptk = tk_qs.data();
            std::vector<QueryResult> res(nq);
            {
                py::gil_scoped_release release;
                if (num_threads <= 1) {
                    for (std::size_t i = 0; i < nq; ++i) {
                        res[i] = self.search(pq + i * dim, ptv[i], ptk[i], k, ef_search, force_mode);
                    }
                } else {
                    #ifdef _OPENMP
                    int prev_t = omp_get_max_threads();
                    omp_set_num_threads(num_threads);
                    #pragma omp parallel for schedule(dynamic, 4)
                    #endif
                    for (std::size_t i = 0; i < nq; ++i) {
                        res[i] = self.search(pq + i * dim, ptv[i], ptk[i], k, ef_search, force_mode);
                    }
                    #ifdef _OPENMP
                    omp_set_num_threads(prev_t);
                    #endif
                }
            }
            return res;
        }, py::arg("q_vecs"), py::arg("tv_qs"), py::arg("tk_qs"),
           py::arg("k") = 10, py::arg("ef_search") = 64, py::arg("force_mode") = -1, py::arg("num_threads") = 1)
        .def("exact_search", [](
            const BiTHNSWIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vec,
            uint32_t tv_q, uint32_t tk_q, uint32_t k) {
            const float* pq = q_vec.data();
            py::gil_scoped_release release;
            return self.exact_search(pq, tv_q, tk_q, k);
        })
        .def("save_snapshot", &BiTHNSWIndex::save_snapshot, py::arg("path"), py::arg("fsync_disk") = true)
        .def("load_snapshot", &BiTHNSWIndex::load_snapshot, py::arg("path"), py::arg("populate_pages") = true)
        .def("open_wal", &BiTHNSWIndex::open_wal, py::arg("wal_path"), py::arg("sync_each_commit") = false)
        .def("begin_wal_tx", &BiTHNSWIndex::begin_wal_tx)
        .def("commit_wal_tx", &BiTHNSWIndex::commit_wal_tx)
        .def("sync_wal", &BiTHNSWIndex::sync_wal)
        .def("close_wal", &BiTHNSWIndex::close_wal)
        .def("replay_wal", &BiTHNSWIndex::replay_wal)
        .def("size", &BiTHNSWIndex::size)
        .def("num_segments", &BiTHNSWIndex::num_segments)
        .def("active_segment_cell_counts", &BiTHNSWIndex::active_segment_cell_counts)
        .def("sealed_segments_cell_counts", &BiTHNSWIndex::sealed_segments_cell_counts)
        .def("corpus_cell_counts", &BiTHNSWIndex::corpus_cell_counts)
        .def("memory_bytes", &BiTHNSWIndex::memory_bytes)
        .def("total_ram_bytes", &BiTHNSWIndex::total_ram_bytes)
        .def("non_graph_ram_bytes", &BiTHNSWIndex::non_graph_ram_bytes)
        .def("verify_identical_state", &BiTHNSWIndex::verify_identical_state)
        .def("mmap_bytes", &BiTHNSWIndex::mmap_bytes)
        .def("offload_fp32_to_mmap", &BiTHNSWIndex::offload_fp32_to_mmap,
             py::arg("dir_path"), py::arg("include_active") = true, py::arg("drop_os_cache") = false)
        .def("drop_fp32_mmap_page_cache", &BiTHNSWIndex::drop_fp32_mmap_page_cache);

    // Lazy-Branch MVCC Engine bindings
    py::class_<TxCommitStats>(m, "TxCommitStats")
        .def_readonly("latency_us", &TxCommitStats::latency_us)
        .def_readonly("tainted_reflections", &TxCommitStats::tainted_reflections)
        .def_readonly("dag_edges_traversed", &TxCommitStats::dag_edges_traversed)
        .def_readonly("split_historical_slice", &TxCommitStats::split_historical_slice)
        .def_readonly("committed", &TxCommitStats::committed)
        .def_readonly("hist_slice_id", &TxCommitStats::hist_slice_id);

    py::class_<MVCCReadResult>(m, "MVCCReadResult")
        .def_readonly("resolved_ids", &MVCCReadResult::resolved_ids)
        .def_readonly("distances", &MVCCReadResult::distances)
        .def_readonly("unpacked_dirty_reflections", &MVCCReadResult::unpacked_dirty_reflections)
        .def_readonly("hot_repair_candidates", &MVCCReadResult::hot_repair_candidates)
        .def_readonly("unpacked_ancestors_evaluated", &MVCCReadResult::unpacked_ancestors_evaluated)
        .def_readonly("raw_items_returned", &MVCCReadResult::raw_items_returned)
        .def_readonly("reflect_items_returned", &MVCCReadResult::reflect_items_returned)
        .def_readonly("refill_rounds", &MVCCReadResult::refill_rounds)
        .def_readonly("index_stats", &MVCCReadResult::index_stats)
        .def_readonly("total_latency_us", &MVCCReadResult::total_latency_us);

    py::class_<TieredPitResult>(m, "TieredPitResult")
        .def_readonly("resolved_ids", &TieredPitResult::resolved_ids)
        .def_readonly("distances", &TieredPitResult::distances)
        .def_readonly("item_tiers", &TieredPitResult::item_tiers)
        .def_readonly("cold_summaries", &TieredPitResult::cold_summaries)
        .def_readonly("matched_anchors", &TieredPitResult::matched_anchors)
        .def_readonly("hot_items_returned", &TieredPitResult::hot_items_returned)
        .def_readonly("cold_items_returned", &TieredPitResult::cold_items_returned)
        .def_readonly("refill_rounds", &TieredPitResult::refill_rounds)
        .def_readonly("hot_latency_us", &TieredPitResult::hot_latency_us)
        .def_readonly("cold_latency_us", &TieredPitResult::cold_latency_us)
        .def_readonly("total_latency_us", &TieredPitResult::total_latency_us);

    py::class_<ConcurrentMixedStats>(m, "ConcurrentMixedStats")
        .def_readonly("elapsed_sec", &ConcurrentMixedStats::elapsed_sec)
        .def_readonly("total_reads", &ConcurrentMixedStats::total_reads)
        .def_readonly("reader_qps", &ConcurrentMixedStats::reader_qps)
        .def_readonly("reader_p50_us", &ConcurrentMixedStats::reader_p50_us)
        .def_readonly("reader_p95_us", &ConcurrentMixedStats::reader_p95_us)
        .def_readonly("reader_p99_us", &ConcurrentMixedStats::reader_p99_us)
        .def_readonly("total_writes", &ConcurrentMixedStats::total_writes)
        .def_readonly("attempted_writes", &ConcurrentMixedStats::attempted_writes)
        .def_readonly("rejected_writes", &ConcurrentMixedStats::rejected_writes)
        .def_readonly("writer_ops_per_s", &ConcurrentMixedStats::writer_ops_per_s)
        .def_readonly("writer_p50_us", &ConcurrentMixedStats::writer_p50_us)
        .def_readonly("writer_p99_us", &ConcurrentMixedStats::writer_p99_us)
        .def_readonly("rollovers_triggered", &ConcurrentMixedStats::rollovers_triggered)
        .def_readonly("tainted_reflections", &ConcurrentMixedStats::tainted_reflections)
        .def_readonly("cardinality_violations", &ConcurrentMixedStats::cardinality_violations);

    py::class_<LazyBranchMVCCEngine>(m, "LazyBranchMVCCEngine")
        .def(py::init<const BiTHNSWConfig&, uint32_t>(), py::arg("idx_cfg"), py::arg("hotness_thresh") = 3)
        .def("bulk_load", [](
            LazyBranchMVCCEngine& self,
            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_e,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_e,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> tiers,
            int num_threads) {
            std::size_t n = static_cast<std::size_t>(ids.shape(0));
            const uint64_t* p_ids = ids.data();
            const float* p_vecs = vecs.data();
            const uint32_t* p_tvs = tv_s.data();
            const uint32_t* p_tve = tv_e.data();
            const uint32_t* p_tks = tk_s.data();
            const uint32_t* p_tke = tk_e.data();
            const uint8_t* p_tiers = tiers.data();
            py::gil_scoped_release release;
            self.bulk_load(p_ids, p_vecs, p_tvs, p_tve, p_tks, p_tke, p_tiers, n, num_threads);
        }, py::arg("ids"), py::arg("vecs"), py::arg("tv_s"), py::arg("tv_e"),
           py::arg("tk_s"), py::arg("tk_e"), py::arg("tiers"), py::arg("num_threads") = 0)
        .def("register_provenance", &LazyBranchMVCCEngine::register_provenance)
        .def("insert_reflection", [](
            LazyBranchMVCCEngine& self,
            uint64_t refl_id,
            py::array_t<float, py::array::c_style | py::array::forcecast> vec,
            uint32_t tv_s, uint32_t tv_e, uint32_t tk_s, uint32_t tk_e,
            const std::vector<uint64_t>& parents) {
            self.insert_reflection(refl_id, vec.data(), tv_s, tv_e, tk_s, tk_e, parents);
        })
        .def("apply_evolve", [](
            LazyBranchMVCCEngine& self,
            uint64_t old_id, uint64_t new_id,
            py::array_t<float, py::array::c_style | py::array::forcecast> new_vec,
            uint32_t tv_chg, uint32_t new_tv_e, uint32_t tk_now) {
            return self.apply_evolve(old_id, new_id, new_vec.data(), tv_chg, new_tv_e, tk_now);
        })
        .def("apply_restate", [](
            LazyBranchMVCCEngine& self,
            uint64_t old_id, uint64_t new_id,
            py::array_t<float, py::array::c_style | py::array::forcecast> corr_vec,
            uint32_t tk_now) {
            return self.apply_restate(old_id, new_id, corr_vec.data(), tk_now);
        })
        .def("materialize_repaired_reflection", [](
            LazyBranchMVCCEngine& self,
            uint64_t dirty_id, uint64_t new_id,
            py::array_t<float, py::array::c_style | py::array::forcecast> new_vec,
            uint32_t branch_tk) {
            self.materialize_repaired_reflection(dirty_id, new_id, new_vec.data(), branch_tk);
        })
        .def("attach_cold_store", &LazyBranchMVCCEngine::attach_cold_store, py::keep_alive<1, 2>())
        .def("insert_raw", [](
            LazyBranchMVCCEngine& self,
            uint64_t id,
            py::array_t<float, py::array::c_style | py::array::forcecast> vec,
            uint32_t tv_s, uint32_t tv_e, uint32_t tk_s, uint32_t tk_e) {
            self.insert_raw(id, vec.data(), tv_s, tv_e, tk_s, tk_e);
        })
        .def("query_pit", [](
            LazyBranchMVCCEngine& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vec,
            uint32_t tv_q, uint32_t tk_q, uint32_t k, uint32_t ef_search, bool adaptive_fetch) {
            const float* pq = q_vec.data();
            py::gil_scoped_release release;
            return self.query_pit(pq, tv_q, tk_q, k, ef_search, adaptive_fetch);
        }, py::arg("q_vec"), py::arg("tv_q"), py::arg("tk_q"),
           py::arg("k") = 10, py::arg("ef_search") = 64, py::arg("adaptive_fetch") = true)
        .def("query_tiered_pit", [](
            LazyBranchMVCCEngine& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vec,
            const std::string& entity_filter,
            uint32_t tv_q, uint32_t tk_q, uint32_t k, uint32_t ef_search) {
            const float* pq = q_vec.data();
            py::gil_scoped_release release;
            return self.query_tiered_pit(pq, entity_filter, tv_q, tk_q, k, ef_search);
        }, py::arg("q_vec"), py::arg("entity_filter") = "", py::arg("tv_q") = 0,
           py::arg("tk_q") = 0, py::arg("k") = 10, py::arg("ef_search") = 64)
        .def("benchmark_concurrent_mixed", [](
            LazyBranchMVCCEngine& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_qs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_qs,
            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> victim_ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> write_vecs,
            uint32_t base_tk_now,
            double restate_ratio,
            double insert_ratio,
            int num_reader_threads,
            uint32_t k,
            uint32_t ef_search,
            double duration_sec) {
            std::size_t nq = static_cast<std::size_t>(q_vecs.shape(0));
            std::size_t nw = static_cast<std::size_t>(write_vecs.shape(0));
            const float* pq = q_vecs.data();
            const uint32_t* ptv = tv_qs.data();
            const uint32_t* ptk = tk_qs.data();
            const uint64_t* pvic = (victim_ids.shape(0) > 0) ? victim_ids.data() : nullptr;
            const float* pw = (nw > 0) ? write_vecs.data() : nullptr;
            py::gil_scoped_release release;
            return self.benchmark_concurrent_mixed(
                pq, ptv, ptk, nq, pvic, pw, nw, base_tk_now,
                restate_ratio, insert_ratio, num_reader_threads, k, ef_search, duration_sec);
        }, py::arg("q_vecs"), py::arg("tv_qs"), py::arg("tk_qs"),
           py::arg("victim_ids"), py::arg("write_vecs"),
           py::arg("base_tk_now"), py::arg("restate_ratio") = 0.20, py::arg("insert_ratio") = 0.50,
           py::arg("num_reader_threads") = 8, py::arg("k") = 10, py::arg("ef_search") = 64,
           py::arg("duration_sec") = 2.0)
        .def("num_records", &LazyBranchMVCCEngine::num_records)
        .def("save_snapshot", &LazyBranchMVCCEngine::save_snapshot, py::arg("base_path"), py::arg("fsync_disk") = true)
        .def("load_snapshot", &LazyBranchMVCCEngine::load_snapshot, py::arg("base_path"), py::arg("populate_pages") = true)
        .def("open_wal", &LazyBranchMVCCEngine::open_wal, py::arg("wal_path"), py::arg("sync_each_commit") = false)
        .def("sync_wal", &LazyBranchMVCCEngine::sync_wal)
        .def("close_wal", &LazyBranchMVCCEngine::close_wal)
        .def("replay_wal", &LazyBranchMVCCEngine::replay_wal, py::arg("wal_path"))
        .def("index", [](LazyBranchMVCCEngine& self) -> BiTHNSWIndex* { return &self.index(); },
             py::return_value_policy::reference_internal);

    // RAP-Compact bindings
    py::class_<RiskAnchorInput>(m, "RiskAnchorInput")
        .def(py::init<>())
        .def_readwrite("source_mem_id", &RiskAnchorInput::source_mem_id)
        .def_readwrite("entity", &RiskAnchorInput::entity)
        .def_readwrite("metric", &RiskAnchorInput::metric)
        .def_readwrite("unit", &RiskAnchorInput::unit)
        .def_readwrite("consequence", &RiskAnchorInput::consequence)
        .def_readwrite("op", &RiskAnchorInput::op)
        .def_readwrite("flags", &RiskAnchorInput::flags)
        .def_readwrite("threshold_val", &RiskAnchorInput::threshold_val)
        .def_readwrite("secondary_val", &RiskAnchorInput::secondary_val)
        .def_readwrite("maturity_tick", &RiskAnchorInput::maturity_tick)
        .def_readwrite("tv_s", &RiskAnchorInput::tv_s)
        .def_readwrite("tv_e", &RiskAnchorInput::tv_e)
        .def_readwrite("tk_s", &RiskAnchorInput::tk_s)
        .def_readwrite("tk_e", &RiskAnchorInput::tk_e);

    py::class_<CompactionStats>(m, "CompactionStats")
        .def_readonly("raw_total_bytes", &CompactionStats::raw_total_bytes)
        .def_readonly("compacted_total_bytes", &CompactionStats::compacted_total_bytes)
        .def_readonly("compression_ratio", &CompactionStats::compression_ratio)
        .def_readonly("space_fraction", &CompactionStats::space_fraction)
        .def_readonly("num_input_records", &CompactionStats::num_input_records)
        .def_readonly("num_clusters", &CompactionStats::num_clusters)
        .def_readonly("num_preserved_anchors", &CompactionStats::num_preserved_anchors);

    py::class_<HybridRAPResult>(m, "HybridRAPResult")
        .def_readonly("cluster_ids", &HybridRAPResult::cluster_ids)
        .def_readonly("member_ids", &HybridRAPResult::member_ids)
        .def_readonly("cluster_distances", &HybridRAPResult::cluster_distances)
        .def_readonly("cluster_summaries", &HybridRAPResult::cluster_summaries)
        .def_readonly("matched_anchors", &HybridRAPResult::matched_anchors)
        .def_readonly("latency_us", &HybridRAPResult::latency_us);

    py::class_<RAPCompactStore>(m, "RAPCompactStore")
        .def(py::init<std::size_t>(), py::arg("orig_dim") = 768)
        .def("compact_batch", [](
            RAPCompactStore& self,
            const std::vector<uint64_t>& mem_ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> embeddings,
            const std::vector<std::string>& raw_texts,
            const std::vector<uint32_t>& tv_s,
            const std::vector<uint32_t>& tv_e,
            const std::vector<uint32_t>& tk_s,
            const std::vector<uint32_t>& tk_e,
            const std::vector<RiskAnchorInput>& anchors,
            double target_budget_ratio) {
            return self.compact_batch(
                mem_ids, embeddings.data(), raw_texts, tv_s, tv_e, tk_s, tk_e, anchors, target_budget_ratio);
        })
        .def("query_anchors_pit", &RAPCompactStore::query_anchors_pit)
        .def("search_hybrid_pit", [](
            const RAPCompactStore& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vec,
            const std::string& entity_filter,
            uint32_t tv_q, uint32_t tk_q, uint32_t top_k) {
            const float* pq = q_vec.data();
            py::gil_scoped_release release;
            return self.search_hybrid_pit(pq, entity_filter, tv_q, tk_q, top_k);
        }, py::arg("q_vec"), py::arg("entity_filter") = "", py::arg("tv_q") = 0, py::arg("tk_q") = 0, py::arg("top_k") = 5)
        .def("save_segment_file", &RAPCompactStore::save_segment_file)
        .def("load_segment_file", &RAPCompactStore::load_segment_file)
        .def("serialized_bytes", &RAPCompactStore::serialized_bytes);

    // Official Upstream Open-Source Library Baseline (nmslib/hnswlib v0.8.0)
    py::class_<OfficialHNSWLibIndex>(m, "OfficialHNSWLibIndex")
        .def(py::init<std::size_t, uint32_t, uint32_t>(),
             py::arg("dim"), py::arg("m") = 16, py::arg("ef_construction") = 96)
        .def("build_batch", [](
            OfficialHNSWLibIndex& self,
            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_e,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_e,
            int num_threads) {
            std::size_t n = static_cast<std::size_t>(ids.shape(0));
            const uint64_t* p_ids = ids.data();
            const float* p_vecs = vecs.data();
            const uint32_t* p_tvs = tv_s.data();
            const uint32_t* p_tve = tv_e.data();
            const uint32_t* p_tks = tk_s.data();
            const uint32_t* p_tke = tk_e.data();
            py::gil_scoped_release release;
            self.build_batch(p_ids, p_vecs, p_tvs, p_tve, p_tks, p_tke, n, num_threads);
        }, py::arg("ids"), py::arg("vecs"), py::arg("tv_s"), py::arg("tv_e"),
           py::arg("tk_s"), py::arg("tk_e"), py::arg("num_threads") = 0)
        .def("update_tx_end", &OfficialHNSWLibIndex::update_tx_end)
        .def("search_batch", [](
            const OfficialHNSWLibIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_qs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_qs,
            uint32_t k, uint32_t ef_search) {
            std::size_t nq = static_cast<std::size_t>(q_vecs.shape(0));
            std::size_t dim = static_cast<std::size_t>(q_vecs.shape(1));
            const float* pq = q_vecs.data();
            const uint32_t* ptv = tv_qs.data();
            const uint32_t* ptk = tk_qs.data();
            std::vector<QueryResult> res(nq);
            {
                py::gil_scoped_release release;
                for (std::size_t i = 0; i < nq; ++i) {
                    res[i] = self.search(pq + i * dim, ptv[i], ptk[i], k, ef_search);
                }
            }
            return res;
        })
        .def("search_acorn1_batch", [](
            const OfficialHNSWLibIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_qs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_qs,
            uint32_t k, uint32_t ef_search) {
            std::size_t nq = static_cast<std::size_t>(q_vecs.shape(0));
            std::size_t dim = static_cast<std::size_t>(q_vecs.shape(1));
            const float* pq = q_vecs.data();
            const uint32_t* ptv = tv_qs.data();
            const uint32_t* ptk = tk_qs.data();
            std::vector<QueryResult> res(nq);
            {
                py::gil_scoped_release release;
                for (std::size_t i = 0; i < nq; ++i) {
                    res[i] = self.search_acorn1(pq + i * dim, ptv[i], ptk[i], k, ef_search);
                }
            }
            return res;
        })
        .def("memory_bytes", &OfficialHNSWLibIndex::memory_bytes);

    // Fair C++ Baselines
    py::class_<HNSWPostIndex>(m, "HNSWPostIndex")
        .def(py::init<std::size_t, uint32_t, uint32_t>(),
             py::arg("dim"), py::arg("m") = 16, py::arg("ef_construction") = 96)
        .def("build_batch", [](
            HNSWPostIndex& self,
            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_e,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_e,
            int num_threads) {
            std::size_t n = static_cast<std::size_t>(ids.shape(0));
            const uint64_t* p_ids = ids.data();
            const float* p_vecs = vecs.data();
            const uint32_t* p_tvs = tv_s.data();
            const uint32_t* p_tve = tv_e.data();
            const uint32_t* p_tks = tk_s.data();
            const uint32_t* p_tke = tk_e.data();
            py::gil_scoped_release release;
            self.build_batch(p_ids, p_vecs, p_tvs, p_tve, p_tks, p_tke, n, num_threads);
        }, py::arg("ids"), py::arg("vecs"), py::arg("tv_s"), py::arg("tv_e"),
           py::arg("tk_s"), py::arg("tk_e"), py::arg("num_threads") = 0)
        .def("update_tx_end", &HNSWPostIndex::update_tx_end)
        .def("search_batch", [](
            const HNSWPostIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_qs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_qs,
            uint32_t k, uint32_t ef_search) {
            std::size_t nq = static_cast<std::size_t>(q_vecs.shape(0));
            std::size_t dim = static_cast<std::size_t>(q_vecs.shape(1));
            const float* pq = q_vecs.data();
            const uint32_t* ptv = tv_qs.data();
            const uint32_t* ptk = tk_qs.data();
            std::vector<QueryResult> res(nq);
            {
                py::gil_scoped_release release;
                for (std::size_t i = 0; i < nq; ++i) {
                    res[i] = self.search(pq + i * dim, ptv[i], ptk[i], k, ef_search);
                }
            }
            return res;
        })
        .def("memory_bytes", &HNSWPostIndex::memory_bytes);

    py::class_<OneDimSegmentGraphIndex>(m, "OneDimSegmentGraphIndex")
        .def(py::init<std::size_t, int, uint32_t, uint32_t>(),
             py::arg("dim"), py::arg("axis_mode") = 0, py::arg("m") = 16, py::arg("ef_construction") = 96)
        .def("build_batch", [](
            OneDimSegmentGraphIndex& self,
            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_e,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_e,
            int num_threads) {
            std::size_t n = static_cast<std::size_t>(ids.shape(0));
            const uint64_t* p_ids = ids.data();
            const float* p_vecs = vecs.data();
            const uint32_t* p_tvs = tv_s.data();
            const uint32_t* p_tve = tv_e.data();
            const uint32_t* p_tks = tk_s.data();
            const uint32_t* p_tke = tk_e.data();
            py::gil_scoped_release release;
            self.build_batch(p_ids, p_vecs, p_tvs, p_tve, p_tks, p_tke, n, num_threads);
        }, py::arg("ids"), py::arg("vecs"), py::arg("tv_s"), py::arg("tv_e"),
           py::arg("tk_s"), py::arg("tk_e"), py::arg("num_threads") = 0)
        .def("update_tx_end", &OneDimSegmentGraphIndex::update_tx_end)
        .def("search_batch", [](
            const OneDimSegmentGraphIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_qs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_qs,
            uint32_t k, uint32_t ef_search) {
            std::size_t nq = static_cast<std::size_t>(q_vecs.shape(0));
            std::size_t dim = static_cast<std::size_t>(q_vecs.shape(1));
            const float* pq = q_vecs.data();
            const uint32_t* ptv = tv_qs.data();
            const uint32_t* ptk = tk_qs.data();
            std::vector<QueryResult> res(nq);
            {
                py::gil_scoped_release release;
                for (std::size_t i = 0; i < nq; ++i) {
                    res[i] = self.search(pq + i * dim, ptv[i], ptk[i], k, ef_search);
                }
            }
            return res;
        })
        .def("memory_bytes", &OneDimSegmentGraphIndex::memory_bytes);

    py::class_<ACORNIndex>(m, "ACORNIndex")
        .def(py::init<std::size_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("dim"), py::arg("m") = 16, py::arg("gamma") = 2, py::arg("ef_construction") = 96)
        .def("build_batch", [](
            ACORNIndex& self,
            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_e,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_e,
            int num_threads) {
            std::size_t n = static_cast<std::size_t>(ids.shape(0));
            const uint64_t* p_ids = ids.data();
            const float* p_vecs = vecs.data();
            const uint32_t* p_tvs = tv_s.data();
            const uint32_t* p_tve = tv_e.data();
            const uint32_t* p_tks = tk_s.data();
            const uint32_t* p_tke = tk_e.data();
            py::gil_scoped_release release;
            self.build_batch(p_ids, p_vecs, p_tvs, p_tve, p_tks, p_tke, n, num_threads);
        }, py::arg("ids"), py::arg("vecs"), py::arg("tv_s"), py::arg("tv_e"),
           py::arg("tk_s"), py::arg("tk_e"), py::arg("num_threads") = 0)
        .def("update_tx_end", &ACORNIndex::update_tx_end)
        .def("search_batch", [](
            const ACORNIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_qs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_qs,
            uint32_t k, uint32_t ef_search) {
            std::size_t nq = static_cast<std::size_t>(q_vecs.shape(0));
            std::size_t dim = static_cast<std::size_t>(q_vecs.shape(1));
            const float* pq = q_vecs.data();
            const uint32_t* ptv = tv_qs.data();
            const uint32_t* ptk = tk_qs.data();
            std::vector<QueryResult> res(nq);
            {
                py::gil_scoped_release release;
                for (std::size_t i = 0; i < nq; ++i) {
                    res[i] = self.search(pq + i * dim, ptv[i], ptk[i], k, ef_search);
                }
            }
            return res;
        })
        .def("memory_bytes", &ACORNIndex::memory_bytes);

    py::class_<GridSQ8ScanIndex>(m, "GridSQ8ScanIndex")
        .def(py::init<std::size_t, uint32_t, uint64_t>(),
             py::arg("dim") = 768, py::arg("grid_bins") = 16, py::arg("initial_seg_cap") = 131072)
        .def("build_batch", [](
            GridSQ8ScanIndex& self,
            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ids,
            py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_e,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_s,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_e,
            int num_threads) {
            std::size_t n = static_cast<std::size_t>(ids.shape(0));
            const uint64_t* p_ids = ids.data();
            const float* p_vecs = vecs.data();
            const uint32_t* p_tvs = tv_s.data();
            const uint32_t* p_tve = tv_e.data();
            const uint32_t* p_tks = tk_s.data();
            const uint32_t* p_tke = tk_e.data();
            py::gil_scoped_release release;
            self.build_batch(p_ids, p_vecs, p_tvs, p_tve, p_tks, p_tke, n, num_threads);
        }, py::arg("ids"), py::arg("vecs"), py::arg("tv_s"), py::arg("tv_e"),
           py::arg("tk_s"), py::arg("tk_e"), py::arg("num_threads") = 0)
        .def("search_batch", [](
            const GridSQ8ScanIndex& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> q_vecs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tv_qs,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tk_qs,
            uint32_t k, uint32_t rerank_pool) {
            std::size_t nq = static_cast<std::size_t>(q_vecs.shape(0));
            std::size_t dim = static_cast<std::size_t>(q_vecs.shape(1));
            const float* pq = q_vecs.data();
            const uint32_t* ptv = tv_qs.data();
            const uint32_t* ptk = tk_qs.data();
            std::vector<QueryResult> res(nq);
            {
                py::gil_scoped_release release;
                for (std::size_t i = 0; i < nq; ++i) {
                    res[i] = self.search(pq + i * dim, ptv[i], ptk[i], k, rerank_pool);
                }
            }
            return res;
        }, py::arg("q_vecs"), py::arg("tv_qs"), py::arg("tk_qs"), py::arg("k") = 10, py::arg("rerank_pool") = 64)
        .def("memory_bytes", &GridSQ8ScanIndex::memory_bytes);
}
