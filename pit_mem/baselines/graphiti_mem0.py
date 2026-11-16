"""Zep/Graphiti Bi-Temporal Property Graph, Mem0 Overwrite, and Naive-GraphRAG Baselines."""

from __future__ import annotations

import sqlite3
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pit_mem_core as pm


class GraphitiBiTempBaseline:
    """
    Faithful architecture reproduction of Zep / Graphiti (Rasmussen et al., arXiv:2501.13956):
      1. Episodic / Relational Edges carry 4 bitemporal columns (t_valid, t_invalid, t_created, t_expired)
         stored in a relational/graph catalog + standard HNSW edge-vector index (post-filtered on AS OF).
      2. Entity / Community Summary Nodes aggregate incident facts, but summary nodes only track
         latest community state (or lack causal tk-branched MVCC invalidation when historic leaf edges restate).
    """

    def __init__(self, dim: int = 768, m: int = 16, ef_c: int = 96):
        self.dim = dim
        self.edge_hnsw = pm.HNSWPostIndex(dim, m, ef_c)
        self.sqlite = sqlite3.connect(":memory:")
        self.sqlite.execute(
            """
            CREATE TABLE graph_edges (
                edge_id INTEGER PRIMARY KEY,
                entity TEXT,
                tier INTEGER,
                t_valid INTEGER,
                t_invalid INTEGER,
                t_created INTEGER,
                t_expired INTEGER
            );
            """
        )
        self.sqlite.execute("CREATE INDEX idx_edges_time ON graph_edges(t_created, t_valid);")
        # Community / entity summaries (unbranched latest-summary overwrite per entity)
        self.community_summary_by_entity: Dict[str, Tuple[int, np.ndarray, int]] = {}
        self.id_to_meta: Dict[int, dict] = {}

    def bulk_load_edges(
        self,
        ids: np.ndarray,
        vecs: np.ndarray,
        tv_s: np.ndarray,
        tv_e: np.ndarray,
        tk_s: np.ndarray,
        tk_e: np.ndarray,
        entities: List[str] | None = None,
        num_threads: int = 0,
    ) -> None:
        self.edge_hnsw.build_batch(ids, vecs, tv_s, tv_e, tk_s, tk_e, num_threads)
        rows = []
        for i in range(len(ids)):
            ent = entities[i] if entities is not None else f"E_{i % 256}"
            rows.append((int(ids[i]), ent, 0, int(tv_s[i]), int(tv_e[i]), int(tk_s[i]), int(tk_e[i])))
            self.id_to_meta[int(ids[i])] = {
                "entity": ent,
                "tier": 0,
                "tv_s": int(tv_s[i]),
                "tv_e": int(tv_e[i]),
                "tk_s": int(tk_s[i]),
                "tk_e": int(tk_e[i]),
            }
        self.sqlite.executemany("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?);", rows)
        self.sqlite.commit()

    def upsert_community_summary(self, entity: str, summary_id: int, summary_vec: np.ndarray, updated_tk: int) -> None:
        """Graphiti updates entity/community summaries in-place without tk-scoped MVCC version branching."""
        self.community_summary_by_entity[entity] = (int(summary_id), summary_vec.astype(np.float32), int(updated_tk))

    def restate_edge_only(self, old_id: int, new_id: int, new_vec: np.ndarray, entity: str, tv_s: int, tv_e: int, tk_now: int) -> None:
        """Graphiti invalidates the contradicted leaf edge (`t_expired = tk_now`), leaving prior reflections untouched unless eagerly regenerated."""
        self.edge_hnsw.update_tx_end(int(old_id), int(tk_now))
        self.sqlite.execute("UPDATE graph_edges SET t_expired = ? WHERE edge_id = ?;", (int(tk_now), int(old_id)))
        arr_id = np.array([new_id], dtype=np.uint64)
        arr_v = new_vec.reshape(1, self.dim).astype(np.float32)
        arr_tvs = np.array([tv_s], dtype=np.uint32)
        arr_tve = np.array([tv_e], dtype=np.uint32)
        arr_tks = np.array([tk_now], dtype=np.uint32)
        arr_tke = np.array([pm.INF_TICK], dtype=np.uint32)
        self.edge_hnsw.build_batch(arr_id, arr_v, arr_tvs, arr_tve, arr_tks, arr_tke, 1)
        self.sqlite.execute(
            "INSERT OR REPLACE INTO graph_edges VALUES (?, ?, 0, ?, ?, ?, ?);",
            (int(new_id), entity, int(tv_s), int(tv_e), int(tk_now), int(pm.INF_TICK)),
        )
        self.sqlite.commit()

    def search_as_of_batch(
        self,
        q_vecs: np.ndarray,
        tv_qs: np.ndarray,
        tk_qs: np.ndarray,
        k: int = 10,
        ef_search: int = 64,
    ) -> Tuple[List[List[int]], np.ndarray]:
        """Hybrid Graphiti search: HNSW post-filtered edges + 1-hop SQLite graph lookup verification."""
        t_lat = np.zeros(len(q_vecs), dtype=np.float64)
        out_ids: List[List[int]] = []
        raw_res = self.edge_hnsw.search_batch(q_vecs, tv_qs, tk_qs, k, ef_search)
        cur = self.sqlite.cursor()
        for i, r in enumerate(raw_res):
            t0 = time.perf_counter()
            if r.ids:
                placeholders = ",".join("?" for _ in r.ids)
                cur.execute(
                    f"SELECT edge_id FROM graph_edges WHERE edge_id IN ({placeholders}) "
                    f"AND t_valid <= ? AND t_invalid > ? AND t_created <= ? AND t_expired > ?;",
                    (*[int(x) for x in r.ids], int(tv_qs[i]), int(tv_qs[i]), int(tk_qs[i]), int(tk_qs[i])),
                )
                verified = [int(row[0]) for row in cur.fetchall()]
            else:
                verified = []
            graph_overhead_us = (time.perf_counter() - t0) * 1e6
            t_lat[i] = r.stats.latency_us + graph_overhead_us
            out_ids.append(verified[:k])
        return out_ids, t_lat

    def memory_bytes(self) -> int:
        cur = self.sqlite.cursor()
        p_cnt = int(cur.execute("PRAGMA page_count;").fetchone()[0])
        p_sz = int(cur.execute("PRAGMA page_size;").fetchone()[0])
        comm_bytes = len(self.community_summary_by_entity) * (self.dim * 4 + 64)
        return int(self.edge_hnsw.memory_bytes()) + p_cnt * p_sz + comm_bytes


class Mem0OverwriteBaseline:
    """Mem0-style mutable latest-state store: overwrites conflicting facts in-place (no transaction-time MVCC)."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.active_entries: Dict[int, Tuple[np.ndarray, str]] = {}

    def put(self, mem_id: int, vec: np.ndarray, tag: str) -> None:
        self.active_entries[int(mem_id)] = (vec.astype(np.float32), tag)

    def overwrite(self, old_id: int, new_id: int, new_vec: np.ndarray, tag: str) -> None:
        self.active_entries.pop(int(old_id), None)
        self.active_entries[int(new_id)] = (new_vec.astype(np.float32), tag)

    def query(self, q_vec: np.ndarray, k: int = 5) -> List[int]:
        if not self.active_entries:
            return []
        ids = list(self.active_entries.keys())
        mat = np.stack([self.active_entries[i][0] for i in ids], axis=0)
        dists = np.sum((mat - q_vec.reshape(1, -1)) ** 2, axis=1)
        order = np.argsort(dists)[:k]
        return [ids[int(j)] for j in order]


class NaiveGraphRAGBaseline:
    """Append-only GraphRAG with global unversioned community summaries (no bi-temporal snapshot filtering)."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.entries: List[Tuple[int, np.ndarray]] = []

    def add(self, mem_id: int, vec: np.ndarray) -> None:
        self.entries.append((int(mem_id), vec.astype(np.float32)))

    def query(self, q_vec: np.ndarray, k: int = 5) -> List[int]:
        if not self.entries:
            return []
        ids = [x[0] for x in self.entries]
        mat = np.stack([x[1] for x in self.entries], axis=0)
        dists = np.sum((mat - q_vec.reshape(1, -1)) ** 2, axis=1)
        order = np.argsort(dists)[:k]
        return [ids[int(j)] for j in order]
