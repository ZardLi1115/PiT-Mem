"""Rigorous Unit & Stress Verification for PVLDB Revision Fixes MW1--MW4."""

from __future__ import annotations

import math
import tempfile
import numpy as np
import pit_mem_core as pm


def _unit_vecs(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


def test_mw1_analytical_router_and_multisegment_recall() -> None:
    rng = np.random.default_rng(1001)
    n, dim = 60_000, 768
    ids = np.arange(1, n + 1, dtype=np.uint64)
    # Realistic clustered semantic manifold (64 topic centroids + intra-cluster variance, matching embedding spaces)
    centers = _unit_vecs(rng, 64, dim)
    c_assign = rng.integers(0, 64, size=n)
    vecs = (centers[c_assign] + rng.standard_normal((n, dim)).astype(np.float32) * (0.45 / math.sqrt(dim))).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    t0 = 1_600_000_000
    tk_s = np.sort(t0 + rng.integers(0, 86400 * 900, size=n, dtype=np.uint32))
    lags = np.where(
        rng.random(n) < 0.80,
        rng.integers(0, 86400 * 14, size=n, dtype=np.uint32),
        rng.integers(86400 * 30, 86400 * 360, size=n, dtype=np.uint32),
    )
    tv_s = np.maximum(t0 - 86400 * 365, tk_s.astype(np.int64) - lags.astype(np.int64)).astype(np.uint32)
    is_open = rng.random(n) < 0.58
    tv_e = np.where(is_open, int(pm.INF_TICK), tv_s + rng.integers(3600, 86400 * 30, size=n, dtype=np.uint32)).astype(np.uint32)
    tk_e = np.full(n, int(pm.INF_TICK), dtype=np.uint32)

    cfg = pm.BiTHNSWConfig()
    cfg.dim = dim
    cfg.ef_construction = 72
    cfg.streaming_mode = True
    cfg.initial_segment_cap = 12288
    cfg.max_segment_cap = 16384

    idx = pm.BiTHNSWIndex(cfg)
    idx.build_batch(ids, vecs, tv_s, tv_e, tk_s, tk_e, 0)
    assert idx.num_segments() >= 4, f"Expected >= 4 segments, got {idx.num_segments()}"

    # Broad multi-segment snapshot (~45% selectivity across 4 segments)
    tv_q_broad = int(np.percentile(tv_s, 82))
    tk_q_broad = int(np.percentile(tk_s, 86))
    vis_broad = np.where((tv_s <= tv_q_broad) & (tv_q_broad < tv_e) & (tk_s <= tk_q_broad) & (tk_q_broad < tk_e))[0]
    nq = 40
    q_anchors = rng.choice(vis_broad, size=nq, replace=False)
    q_vecs = (vecs[q_anchors] + rng.standard_normal((nq, dim)).astype(np.float32) * (0.50 / math.sqrt(dim))).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-12
    tv_qs = np.full(nq, tv_q_broad, dtype=np.uint32)
    tk_qs = np.full(nq, tk_q_broad, dtype=np.uint32)

    gts = [set(idx.exact_search(q_vecs[i], tv_q_broad, tk_q_broad, 10).ids) for i in range(nq)]
    res_96 = idx.search_batch(q_vecs, tv_qs, tk_qs, 10, 96, -1, 1)
    res_384 = idx.search_batch(q_vecs, tv_qs, tk_qs, 10, 384, 0, 1)

    rec_96 = float(np.mean([len(set(res_96[i].ids) & gts[i]) / 10.0 for i in range(nq)]))
    rec_384 = float(np.mean([len(set(res_384[i].ids) & gts[i]) / 10.0 for i in range(nq)]))
    graph_pct = float(np.mean([1.0 if r.stats.graph_cells > 0 else 0.0 for r in res_96])) * 100.0
    print(f"[MW1 Multi-Seg Graph] segs={idx.num_segments()}, graph_pct(ef=96)={graph_pct:.1f}%, R@10(ef=96)={rec_96:.3f}, R@10(ef=384,graph)={rec_384:.3f}")
    assert graph_pct >= 95.0
    assert rec_96 >= 0.90 and rec_384 >= 0.96

    # Sparse snapshot -> analytical router selects scan cleanly with ~100% recall
    tv_q_sp = int(np.percentile(tv_s, 18))
    tk_q_sp = int(np.percentile(tk_s, 22))
    tv_qs_sp = np.full(nq, tv_q_sp, dtype=np.uint32)
    tk_qs_sp = np.full(nq, tk_q_sp, dtype=np.uint32)
    gts_sp = [set(idx.exact_search(q_vecs[i], tv_q_sp, tk_q_sp, 10).ids) for i in range(nq)]
    res_sp = idx.search_batch(q_vecs, tv_qs_sp, tk_qs_sp, 10, 96, -1, 1)
    rec_sp = float(np.mean([len(set(res_sp[i].ids) & gts_sp[i]) / max(1, len(gts_sp[i])) for i in range(nq)]))
    print(f"[MW1 Sparse Router] R@10={rec_sp:.4f}")
    assert rec_sp >= 0.98


def test_mw2_nvme_mmap_offload_and_concurrent_stress() -> None:
    rng = np.random.default_rng(2002)
    n, dim = 20_000, 768
    ids = np.arange(1, n + 1, dtype=np.uint64)
    vecs = _unit_vecs(rng, n, dim)
    t0 = 1_650_000_000
    tk_s = np.sort(t0 + rng.integers(0, 86400 * 365, size=n, dtype=np.uint32))
    tv_s = (tk_s - rng.integers(0, 86400 * 15, size=n, dtype=np.uint32)).astype(np.uint32)
    tv_e = np.full(n, int(pm.INF_TICK), dtype=np.uint32)
    tk_e = np.full(n, int(pm.INF_TICK), dtype=np.uint32)

    cfg = pm.BiTHNSWConfig()
    cfg.dim = dim
    cfg.ef_construction = 64
    cfg.streaming_mode = True
    cfg.initial_segment_cap = 8192
    cfg.max_segment_cap = 8192

    idx = pm.BiTHNSWIndex(cfg)
    idx.build_batch(ids, vecs, tv_s, tv_e, tk_s, tk_e, 0)
    ram_before_b_per_rec = idx.memory_bytes() / float(n)

    q_vecs = _unit_vecs(rng, 25, dim)
    tv_qs = np.full(25, int(np.percentile(tv_s, 80)), dtype=np.uint32)
    tk_qs = np.full(25, int(np.percentile(tk_s, 85)), dtype=np.uint32)
    res_ram = idx.search_batch(q_vecs, tv_qs, tk_qs, 10, 96, -1, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        st_off = idx.offload_fp32_to_mmap(tmpdir, True, True)
        ram_after_b_per_rec = idx.memory_bytes() / float(n)
        mmap_b_per_rec = idx.mmap_bytes() / float(n)
        idx.drop_fp32_mmap_page_cache()
        res_mmap = idx.search_batch(q_vecs, tv_qs, tk_qs, 10, 96, -1, 1)

        print(
            f"[MW2 NVMe Offload] RAM before={ram_before_b_per_rec:.1f} B/rec -> "
            f"after={ram_after_b_per_rec:.1f} B/rec, mmap={mmap_b_per_rec:.1f} B/rec, "
            f"offload_bw={st_off.throughput_mb_s:.0f} MB/s"
        )
        assert ram_before_b_per_rec - ram_after_b_per_rec >= 3000.0
        assert ram_after_b_per_rec < 1350.0
        assert abs(mmap_b_per_rec - 3072.0) < 1.0
        for r1, r2 in zip(res_ram, res_mmap):
            assert list(r1.ids) == list(r2.ids)
            assert np.allclose(r1.distances, r2.distances, atol=1e-6)

    # Concurrent Multi-Threaded Reader + Writer/Restate/Rollover Stress Test
    cfg_c = pm.BiTHNSWConfig()
    cfg_c.dim = 128
    cfg_c.ef_construction = 56
    cfg_c.streaming_mode = True
    cfg_c.initial_segment_cap = 4096
    cfg_c.max_segment_cap = 6144
    mvcc = pm.LazyBranchMVCCEngine(cfg_c, hotness_thresh=3)

    nc = 10_000
    c_ids = np.arange(1, nc + 1, dtype=np.uint64)
    c_vecs = _unit_vecs(rng, nc, 128)
    c_tks = np.sort(t0 + rng.integers(0, 86400 * 180, size=nc, dtype=np.uint32))
    c_tvs = (c_tks - 3600).astype(np.uint32)
    c_tve = np.full(nc, int(pm.INF_TICK), dtype=np.uint32)
    c_tke = np.full(nc, int(pm.INF_TICK), dtype=np.uint32)
    c_tiers = np.zeros(nc, dtype=np.uint8)
    c_tiers[-1000:] = 1
    mvcc.bulk_load(c_ids, c_vecs, c_tvs, c_tve, c_tks, c_tke, c_tiers, 0)
    for r_i in range(1000):
        mvcc.register_provenance(int(c_ids[9000 + r_i]), [int(c_ids[r_i * 5 + j]) for j in range(5)])

    cq_vecs = _unit_vecs(rng, 64, 128)
    cq_tvs = np.full(64, int(np.percentile(c_tvs, 75)), dtype=np.uint32)
    cq_tks = np.full(64, int(np.max(c_tks)), dtype=np.uint32)
    nw = 3500
    vic_ids = rng.choice(c_ids[:9000], size=nw, replace=True).astype(np.uint64)
    w_vecs = _unit_vecs(rng, nw, 128)

    cst = mvcc.benchmark_concurrent_mixed(
        cq_vecs, cq_tvs, cq_tks, vic_ids, w_vecs,
        int(np.max(c_tks)) + 100, 0.25, 0.55, 6, 10, 64, 1.3,
    )
    print(
        f"[MW2 Concurrent Mixed] reads={cst.total_reads} ({cst.reader_qps:.0f} QPS, P99={cst.reader_p99_us:.0f} us), "
        f"writes={cst.total_writes} ({cst.writer_ops_per_s:.0f} ops/s), rollovers={cst.rollovers_triggered}, "
        f"tainted={cst.tainted_reflections}, violations={cst.cardinality_violations}"
    )
    assert cst.cardinality_violations == 0
    assert cst.total_reads > 200 and cst.total_writes > 200
    assert cst.rollovers_triggered >= 1


def test_mw3_clustered_3hop_adaptive_fetch_and_tiered_pit() -> None:
    rng = np.random.default_rng(3003)
    dim = 128
    q = _unit_vecs(rng, 1, dim)[0]

    cfg = pm.BiTHNSWConfig()
    cfg.dim = dim
    cfg.ef_construction = 80
    cfg.streaming_mode = False
    mvcc = pm.LazyBranchMVCCEngine(cfg, hotness_thresh=3)

    # Create:
    # - 28 RAW leaf records near q initially, later restated to orthogonal vectors!
    # - 28 3-hop reflection chains (RAW -> L1 -> L2 -> L3) whose L3 vectors sit VERY close to q (ranks 1..28)!
    # - 15 clean RAW records at medium-close distance to q (ranks 29..43, the true clean Top-10!)
    # - 200 background clean RAW records far from q
    ids, vecs, tvs, tve, tks, tke, tiers = [], [], [], [], [], [], []

    def _near_q(noise_scale: float) -> np.ndarray:
        v = q + rng.standard_normal(dim).astype(np.float32) * (noise_scale / math.sqrt(dim))
        return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

    n_burst = 28
    for i in range(n_burst):
        # RAW leaf (1..28)
        ids.append(1 + i); vecs.append(_near_q(0.08)); tvs.append(100); tve.append(int(pm.INF_TICK)); tks.append(100); tke.append(int(pm.INF_TICK)); tiers.append(0)
        # L1 (1001..1028), L2 (2001..2028), L3 (3001..3028) - L3 sits closest to q!
        ids.append(1001 + i); vecs.append(_near_q(1.20)); tvs.append(100); tve.append(int(pm.INF_TICK)); tks.append(110); tke.append(int(pm.INF_TICK)); tiers.append(1)
        ids.append(2001 + i); vecs.append(_near_q(1.20)); tvs.append(100); tve.append(int(pm.INF_TICK)); tks.append(120); tke.append(int(pm.INF_TICK)); tiers.append(1)
        ids.append(3001 + i); vecs.append(_near_q(0.05)); tvs.append(100); tve.append(int(pm.INF_TICK)); tks.append(130); tke.append(int(pm.INF_TICK)); tiers.append(1)

    clean_scored = []
    for j in range(15):
        cid = 5000 + j
        cv = _near_q(0.22 + 0.01 * j)
        clean_scored.append((float(np.sum((cv - q) ** 2)), cid))
        ids.append(cid)
        vecs.append(cv)
        tvs.append(100); tve.append(int(pm.INF_TICK)); tks.append(100); tke.append(int(pm.INF_TICK)); tiers.append(0)
    clean_scored.sort(key=lambda x: x[0])
    expected_clean_top10 = [cid for _, cid in clean_scored[:10]]

    for b in range(200):
        ids.append(8000 + b)
        vecs.append(_unit_vecs(rng, 1, dim)[0])
        tvs.append(100); tve.append(int(pm.INF_TICK)); tks.append(100); tke.append(int(pm.INF_TICK)); tiers.append(0)

    mvcc.bulk_load(
        np.array(ids, dtype=np.uint64),
        np.asarray(vecs, dtype=np.float32),
        np.array(tvs, dtype=np.uint32),
        np.array(tve, dtype=np.uint32),
        np.array(tks, dtype=np.uint32),
        np.array(tke, dtype=np.uint32),
        np.array(tiers, dtype=np.uint8),
        0,
    )
    for i in range(n_burst):
        mvcc.register_provenance(1001 + i, [1 + i])
        mvcc.register_provenance(2001 + i, [1001 + i])
        mvcc.register_provenance(3001 + i, [2001 + i])

    # Clustered restatement burst across all 28 leaves -> restated leaves move far from q!
    for i in range(n_burst):
        far_v = (-q + rng.standard_normal(dim).astype(np.float32) * 0.05).astype(np.float32)
        far_v /= np.linalg.norm(far_v) + 1e-12
        st = mvcc.apply_restate(1 + i, 90_000 + i, far_v, 500 + i)
        assert st.tainted_reflections == 3, f"Expected 3-hop taint=3, got {st.tainted_reflections}"

    r_static = mvcc.query_pit(q, 200, 1000, 10, 128, False)
    r_adapt = mvcc.query_pit(q, 200, 1000, 10, 128, True)
    gt_set = set(expected_clean_top10)
    rec_static = len(set(r_static.resolved_ids) & gt_set) / 10.0
    rec_adapt = len(set(r_adapt.resolved_ids) & gt_set) / 10.0
    print(
        f"[MW3 3-Hop Burst] static_recall={rec_static:.2f} (rounds={r_static.refill_rounds}), "
        f"adaptive_recall={rec_adapt:.2f} (rounds={r_adapt.refill_rounds})"
    )
    assert rec_static < 0.30
    assert rec_adapt == 1.0 and r_adapt.refill_rounds >= 1
    assert len(r_adapt.resolved_ids) == 10

    # Unified Hot + Cold Tiered Search (`query_tiered_pit`)
    cold = pm.RAPCompactStore(dim)
    cold_ids = [77_001, 77_002, 77_003, 77_004]
    cold_embs = np.vstack([
        _near_q(0.02),
        _near_q(0.03),
        _unit_vecs(rng, 1, dim)[0],
        _unit_vecs(rng, 1, dim)[0],
    ]).astype(np.float32)
    cold_texts = [
        "Cold covenant senior leverage <= 3.50x EBITDA",
        "Cold covenant interest coverage >= 2.25x amendment",
        "Unrelated cold macro note A",
        "Unrelated cold macro note B",
    ]
    anc = pm.RiskAnchorInput()
    anc.source_mem_id = 77_001
    anc.entity = "FIRM_TIER"
    anc.metric = "net_leverage"
    anc.unit = "x"
    anc.consequence = "cross-default"
    anc.threshold_val = 3.50
    anc.tv_s, anc.tv_e, anc.tk_s, anc.tk_e = 100, int(pm.INF_TICK), 100, int(pm.INF_TICK)
    cold.compact_batch(
        cold_ids, cold_embs, cold_texts,
        [100] * 4, [int(pm.INF_TICK)] * 4,
        [100] * 4, [int(pm.INF_TICK)] * 4,
        [anc], 0.50,
    )
    mvcc.attach_cold_store(cold)
    tr = mvcc.query_tiered_pit(q, "FIRM_TIER", 200, 1000, 10, 128)
    print(
        f"[MW3 Tiered PiT] hot={tr.hot_items_returned}, cold={tr.cold_items_returned}, "
        f"anchors={len(tr.matched_anchors)}, top_id={tr.resolved_ids[0]} (dist={tr.distances[0]:.4f})"
    )
    assert len(tr.resolved_ids) == 10
    assert tr.cold_items_returned >= 1 and 77_001 in list(tr.resolved_ids)
    assert len(tr.matched_anchors) == 1 and abs(tr.matched_anchors[0].threshold_val - 3.50) < 1e-6


def test_mw4_streaming_cell_skew_elimination() -> None:
    rng = np.random.default_rng(4004)
    n, dim = 50_000, 64
    ids = np.arange(1, n + 1, dtype=np.uint64)
    vecs = _unit_vecs(rng, n, dim)
    t_min = 1_520_000_000
    span = 8 * 365 * 86400
    prog = np.linspace(0.0, 1.0, n, endpoint=False)
    tk_s = np.sort((t_min + (prog * span).astype(np.int64)).astype(np.uint32))
    lags = np.where(
        prog < 0.26,
        rng.integers(0, 14 * 86400, size=n, dtype=np.uint32),
        np.where(
            rng.random(n) < 0.75,
            rng.integers(0, 95 * 86400, size=n, dtype=np.uint32),
            rng.integers(95 * 86400, 480 * 86400, size=n, dtype=np.uint32),
        ),
    )
    tv_s = np.maximum(t_min, tk_s.astype(np.int64) - lags.astype(np.int64)).astype(np.uint32)
    tv_e = np.full(n, int(pm.INF_TICK), dtype=np.uint32)
    tk_e = np.full(n, int(pm.INF_TICK), dtype=np.uint32)

    cfg = pm.BiTHNSWConfig()
    cfg.dim = dim
    cfg.ef_construction = 48
    cfg.streaming_mode = True
    cfg.initial_segment_cap = 8192
    cfg.max_segment_cap = 16384
    idx = pm.BiTHNSWIndex(cfg)
    for b0 in range(0, n, 2500):
        b1 = min(n, b0 + 2500)
        idx.build_batch(ids[b0:b1], vecs[b0:b1], tv_s[b0:b1], tv_e[b0:b1], tk_s[b0:b1], tk_e[b0:b1], 0)

    cc_sealed = np.array(idx.sealed_segments_cell_counts(), dtype=np.float64)
    cc_tail = np.array(idx.active_segment_cell_counts(), dtype=np.float64)
    cc_corp = np.array(idx.corpus_cell_counts(), dtype=np.float64)
    sealed_max = float(np.max(cc_sealed) / max(1.0, np.sum(cc_sealed))) * 100.0
    tail_max = float(np.max(cc_tail) / max(1.0, np.sum(cc_tail))) * 100.0
    corp_max = float(np.max(cc_corp) / max(1.0, np.sum(cc_corp))) * 100.0
    c15_15 = float(cc_tail[-1] / max(1.0, np.sum(cc_tail))) * 100.0
    print(
        f"[MW4 Streaming Skew] segs={idx.num_segments()}, corp_max={corp_max:.2f}%, "
        f"sealed_max={sealed_max:.2f}%, tail_max={tail_max:.2f}%, tail_(15,15)={c15_15:.2f}%"
    )
    assert c15_15 == 0.0
    assert sealed_max < 10.0 and corp_max < 10.0
    assert tail_max < 20.0


def test_post_codex_engine_wal_cold_restate_and_preconditions() -> None:
    rng = np.random.default_rng(5005)
    dim = 64
    q = _unit_vecs(rng, 1, dim)[0]

    def _near(base: np.ndarray, scale: float) -> np.ndarray:
        v = base + rng.standard_normal(dim).astype(np.float32) * (scale / math.sqrt(dim))
        return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

    # 1) P1.3: Multi-segment graph search with small ef_search=24 (< 32)
    cfg_ms = pm.BiTHNSWConfig()
    cfg_ms.dim = dim
    cfg_ms.ef_construction = 48
    cfg_ms.streaming_mode = True
    cfg_ms.initial_segment_cap = 4096
    cfg_ms.max_segment_cap = 4096
    idx_ms = pm.BiTHNSWIndex(cfg_ms)
    n_ms = 10_000
    ms_ids = np.arange(1, n_ms + 1, dtype=np.uint64)
    ms_vecs = _unit_vecs(rng, n_ms, dim)
    ms_tks = np.sort(1_000_000 + np.arange(n_ms, dtype=np.uint32) * 10)
    ms_tvs = (ms_tks - 50).astype(np.uint32)
    ms_tve = np.full(n_ms, int(pm.INF_TICK), dtype=np.uint32)
    ms_tke = np.full(n_ms, int(pm.INF_TICK), dtype=np.uint32)
    idx_ms.build_batch(ms_ids, ms_vecs, ms_tvs, ms_tve, ms_tks, ms_tke, 0)
    assert idx_ms.num_segments() >= 3
    r_ef24 = idx_ms.search(q, int(ms_tvs[-1]), int(ms_tks[-1]), 10, 24, 0)
    assert len(r_ef24.ids) == 10

    # 2) P0.1 + P0.2 + P1.2: Full-engine Provenance WAL + Torn Tail + Cold Restatement + Preconditions
    cfg = pm.BiTHNSWConfig()
    cfg.dim = dim
    cfg.ef_construction = 64
    cfg.streaming_mode = False
    mvcc = pm.LazyBranchMVCCEngine(cfg, hotness_thresh=2)

    v_raw1 = _near(q, 0.06)
    v_raw2 = _near(q, 0.09)
    v_refl_dirty = _near(q, 0.02)
    v_restate = _near(-q, 0.05)
    v_refl_clean = _near(q, 0.03)
    v_evolve = _near(q, 0.11)

    mvcc.bulk_load(
        np.array([10, 20], dtype=np.uint64),
        np.vstack([v_raw1, v_raw2]).astype(np.float32),
        np.array([100, 100], dtype=np.uint32),
        np.array([int(pm.INF_TICK), 1000], dtype=np.uint32),
        np.array([100, 100], dtype=np.uint32),
        np.array([int(pm.INF_TICK), int(pm.INF_TICK)], dtype=np.uint32),
        np.array([0, 0], dtype=np.uint8),
        0,
    )

    cold = pm.RAPCompactStore(dim)
    v_cold = _near(q, 0.01)
    v_cold_restate = _near(q, 0.04)
    anc = pm.RiskAnchorInput()
    anc.source_mem_id = 77_001
    anc.entity = "ACME_COLD"
    anc.metric = "leverage"
    anc.unit = "x"
    anc.consequence = "default"
    anc.threshold_val = 4.25
    anc.tv_s, anc.tv_e, anc.tk_s, anc.tk_e = 100, int(pm.INF_TICK), 100, int(pm.INF_TICK)
    cold.compact_batch(
        [77_001, 77_002],
        np.vstack([v_cold, _unit_vecs(rng, 1, dim)[0]]).astype(np.float32),
        ["ACME leverage <= 4.25x", "Other note"],
        [100, 100], [int(pm.INF_TICK), int(pm.INF_TICK)],
        [100, 100], [int(pm.INF_TICK), int(pm.INF_TICK)],
        [anc], 0.50,
    )
    mvcc.attach_cold_store(cold)

    with tempfile.TemporaryDirectory() as tmpdir:
        snap_base = f"{tmpdir}/eng_snap"
        wal_path = f"{tmpdir}/eng.pwal"
        mvcc.save_snapshot(snap_base, True)
        mvcc.open_wal(wal_path, True)

        mvcc.insert_reflection(100, v_refl_dirty, 100, int(pm.INF_TICK), 150, int(pm.INF_TICK), [10, 20])
        st_ev = mvcc.apply_evolve(20, 21, v_evolve, 500, 1000, 300)
        assert st_ev.split_historical_slice
        # P1.2 precondition: invalid evolve split outside (tv_s, tv_e) or on already closed record 20 must be rejected
        st_bad_ev = mvcc.apply_evolve(20, 22, v_evolve, 600, 1000, 350)
        assert st_bad_ev.tainted_reflections == 0 and not st_bad_ev.split_historical_slice
        st_bad_ev2 = mvcc.apply_evolve(21, 23, v_evolve, 1200, 1500, 350)
        assert not st_bad_ev2.split_historical_slice

        # P0.2: Restate hot record 10 -> 11 at tk=400 AND restate cold-only record 77_001 -> 77_101 at tk=450
        st_rs = mvcc.apply_restate(10, 11, v_restate, 400)
        assert st_rs.tainted_reflections >= 1
        # P1.2: Re-restating already closed 10 or reusing existing new_id 11 must be rejected cleanly
        st_dup = mvcc.apply_restate(10, 12, v_restate, 410)
        assert st_dup.tainted_reflections == 0
        st_reuse = mvcc.apply_restate(21, 11, v_restate, 410)
        assert st_reuse.tainted_reflections == 0

        mvcc.apply_restate(77_001, 77_101, v_cold_restate, 450)
        mvcc.materialize_repaired_reflection(100, 101, v_refl_clean, 600)
        mvcc.close_wal()

        # Verify cold-only restatement before & after tk=450
        tr_pre = mvcc.query_tiered_pit(q, "ACME_COLD", 250, 420, 5, 64)
        tr_post = mvcc.query_tiered_pit(q, "ACME_COLD", 250, 500, 5, 64)
        assert 77_001 in list(tr_pre.resolved_ids) and len(tr_pre.matched_anchors) == 1
        assert 77_001 not in list(tr_post.resolved_ids) and len(tr_post.matched_anchors) == 0
        assert 77_101 in list(tr_post.resolved_ids)

        # Simulate mid-transaction crash by appending uncommitted BEGIN_TX + torn garbage bytes to wal_path
        with open(wal_path, "ab") as wf:
            import struct
            # Valid magic "PVWL" + op=10 (BEGIN_TX) followed by torn partial frame
            wf.write(b"LWVP\x0a\x00\x00\x00" + b"\x99" * 29)

        # Recover into a brand-new engine from snapshot + WAL replay!
        rec_eng = pm.LazyBranchMVCCEngine(cfg, hotness_thresh=2)
        rec_eng.attach_cold_store(cold)
        rec_eng.load_snapshot(snap_base, True)
        rep_st = rec_eng.replay_wal(wal_path)
        assert rep_st.records_processed >= 5

        for tv_q, tk_q in [(250, 200), (250, 420), (600, 500), (600, 700)]:
            orig_r = mvcc.query_tiered_pit(q, "ACME_COLD", tv_q, tk_q, 5, 64)
            rec_r = rec_eng.query_tiered_pit(q, "ACME_COLD", tv_q, tk_q, 5, 64)
            assert list(orig_r.resolved_ids) == list(rec_r.resolved_ids), (
                f"Mismatch at ({tv_q},{tk_q}): {orig_r.resolved_ids} vs {rec_r.resolved_ids}"
            )
            assert len(orig_r.matched_anchors) == len(rec_r.matched_anchors)
        print(
            f"[Post-Codex Engine WAL + Cold Restate + Preconditions] replayed={rep_st.records_processed} ops, "
            f"100% snapshot & tiered parity across crash-truncated WAL!"
        )


def test_router_hardware_auto_calibration():
    cfg = pm.BiTHNSWConfig()
    cfg.auto_calibrate_router = True
    idx = pm.BiTHNSWIndex(cfg)
    cal = idx.calibrate_router_hardware(47_185_920)
    assert 0.04 <= cal.router_gamma_ts <= 0.45
    assert 400.0 <= cal.router_gamma_hub <= 2400.0
    assert cal.router_gamma_llc > 0.0 and cal.router_gamma_dim > 0.0
    assert 100_000.0 <= cal.router_llc_knee <= 2_500_000.0
    print(
        f"[Router Auto-Calibration] gamma_0={cal.router_gamma_0:.1f}, gamma_ts={cal.router_gamma_ts:.3f}, "
        f"gamma_hub={cal.router_gamma_hub:.1f}, gamma_llc={cal.router_gamma_llc:.2f}, "
        f"gamma_dim={cal.router_gamma_dim:.2f}, llc_knee={cal.router_llc_knee:.0f}"
    )


def test_official_upstream_baselines_and_bit_exact_state():
    import tempfile
    rng = np.random.default_rng(123)
    n, d = 1500, 64
    ids = np.arange(1, n + 1, dtype=np.uint64)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    tvs = rng.integers(100, 1000, size=n, dtype=np.uint32)
    tve = tvs + rng.integers(80, 900, size=n, dtype=np.uint32)
    tks = rng.integers(100, 1000, size=n, dtype=np.uint32)
    tke = np.full(n, int(pm.INF_TICK), dtype=np.uint32)

    ho = pm.OfficialHNSWLibIndex(d, 16, 64)
    ho.build_batch(ids, vecs, tvs, tve, tks, tke, 0)
    qv = vecs[:8].copy()
    tvq = np.full(8, 500, dtype=np.uint32)
    tkq = np.full(8, 950, dtype=np.uint32)
    r_h = ho.search_batch(qv, tvq, tkq, 10, 96)
    r_a = ho.search_acorn1_batch(qv, tvq, tkq, 10, 96)
    assert len(r_h) == 8 and len(r_a) == 8 and ho.memory_bytes() > n * d * 4

    cfg = pm.BiTHNSWConfig()
    cfg.dim = d
    idx = pm.BiTHNSWIndex(cfg)
    idx.build_batch(ids, vecs, tvs, tve, tks, tke, 0)
    assert idx.total_ram_bytes() > idx.memory_bytes()
    assert idx.non_graph_ram_bytes() < idx.total_ram_bytes()

    with tempfile.TemporaryDirectory() as tmpd:
        sf = f"{tmpd}/s.snap"
        wf = f"{tmpd}/s.wal"
        idx.save_snapshot(sf, True)
        idx.open_wal(wf, False)
        idx.begin_wal_tx()
        idx.insert_single(999_001, vecs[0], 200, int(pm.INF_TICK), 1200, int(pm.INF_TICK))
        idx.update_tx_end(1, 1250)
        idx.commit_wal_tx()
        idx.sync_wal()
        idx.close_wal()
        with open(wf, "ab") as fw:
            fw.write(b"LAWP\x0a\x00\x00\x00" + b"\x55" * 17)

        rec = pm.BiTHNSWIndex(cfg)
        rec.load_snapshot(sf, True)
        rec.replay_wal(wf)
        assert rec.verify_identical_state(idx)
    print("[Official hnswlib + ACORN-1 + Bit-Exact Torn-WAL State Verification] PASSED!")


if __name__ == "__main__":
    test_mw1_analytical_router_and_multisegment_recall()
    test_mw2_nvme_mmap_offload_and_concurrent_stress()
    test_mw3_clustered_3hop_adaptive_fetch_and_tiered_pit()
    test_mw4_streaming_cell_skew_elimination()
    test_post_codex_engine_wal_cold_restate_and_preconditions()
    test_router_hardware_auto_calibration()
    test_official_upstream_baselines_and_bit_exact_state()
    print("\nALL MW1--MW4 + POST-CODEX UNIT & CONCURRENCY STRESS TESTS PASSED 100%!")
