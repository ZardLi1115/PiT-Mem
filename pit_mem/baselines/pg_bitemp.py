"""PostgreSQL + pgvector 0.8+ Bi-Temporal Baseline (Warm Buffer Cache, Prepared Statements, Covering B-Tree + Iterative HNSW)."""

from __future__ import annotations

import io
import os
import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np
import psycopg


def ensure_local_postgres_running(db_name: str = "pitmem_bench") -> str:
    """Start local PostgreSQL service in Ubuntu/Debian container (Colab or local VM), tune shared_buffers, and enable pgvector."""
    mem_gb = (
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        if hasattr(os, "sysconf")
        else 8.0
    )
    subprocess.run(
        "sudo service postgresql start >/dev/null 2>&1 || sudo pg_ctlcluster 17 main start >/dev/null 2>&1 || sudo pg_ctlcluster 16 main start >/dev/null 2>&1 || true",
        shell=True,
        check=False,
    )
    if mem_gb >= 24.0:
        sb_target = "12GB" if mem_gb >= 40.0 else "8GB"
        cur_sb = subprocess.run(
            "sudo -u postgres psql -tc 'SHOW shared_buffers;'",
            shell=True, capture_output=True, text=True, check=False,
        ).stdout.strip()
        if cur_sb != sb_target:
            subprocess.run(
                f"sudo -u postgres psql -c \"ALTER SYSTEM SET shared_buffers = '{sb_target}';\" >/dev/null 2>&1 && "
                "sudo -u postgres psql -c \"ALTER SYSTEM SET effective_cache_size = '36GB';\" >/dev/null 2>&1 && "
                "sudo -u postgres psql -c \"ALTER SYSTEM SET max_wal_size = '8GB';\" >/dev/null 2>&1 && "
                "(sudo service postgresql restart >/dev/null 2>&1 || sudo pg_ctlcluster 17 main restart >/dev/null 2>&1 || sudo pg_ctlcluster 16 main restart >/dev/null 2>&1 || true)",
                shell=True,
                check=False,
            )
    subprocess.run(
        "sudo -u postgres psql -c \"ALTER USER postgres WITH PASSWORD 'postgres';\" >/dev/null 2>&1 || true",
        shell=True,
        check=False,
    )
    subprocess.run(
        f"sudo -u postgres psql -tc \"SELECT 1 FROM pg_database WHERE datname='{db_name}'\" | grep -q 1 || sudo -u postgres createdb {db_name}",
        shell=True,
        check=False,
    )
    subprocess.run(
        f"sudo -u postgres psql -d {db_name} -c 'CREATE EXTENSION IF NOT EXISTS vector;' >/dev/null 2>&1",
        shell=True,
        check=False,
    )
    return f"postgresql://postgres:postgres@127.0.0.1:5432/{db_name}"


class PGBiTempBaseline:
    """Tuned PostgreSQL + pgvector (`halfvec` FP16 HNSW + Covering B-Tree) with VACUUM ANALYZE, Warm Cache & Prepared SQL."""

    def __init__(self, dim: int = 768, table_name: str = "bitemp_memories", dsn: Optional[str] = None):
        self.dim = dim
        self.table = table_name
        self.dsn = dsn or ensure_local_postgres_running()
        self.conn = psycopg.connect(self.dsn, autocommit=True)

    def build_batch(
        self,
        ids: np.ndarray,
        vecs: np.ndarray,
        tv_s: np.ndarray,
        tv_e: np.ndarray,
        tk_s: np.ndarray,
        tk_e: np.ndarray,
        m: int = 16,
        ef_construction: int = 64,
    ) -> float:
        t0 = time.perf_counter()
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table};")
            cur.execute(
                f"""
                CREATE UNLOGGED TABLE {self.table} (
                    mem_id BIGINT PRIMARY KEY,
                    tv_s BIGINT NOT NULL,
                    tv_e BIGINT NOT NULL,
                    tk_s BIGINT NOT NULL,
                    tk_e BIGINT NOT NULL,
                    embedding halfvec({self.dim}) NOT NULL
                );
                """
            )
            mem_gb = (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
                if hasattr(os, "sysconf")
                else 8.0
            )
            maint_mem = "8GB" if mem_gb >= 32.0 else ("4GB" if mem_gb >= 20.0 else "512MB")
            par_workers = 10 if mem_gb >= 32.0 else (8 if mem_gb >= 20.0 else 2)
            cur.execute(f"SET maintenance_work_mem = '{maint_mem}';")
            cur.execute(f"SET max_parallel_maintenance_workers = {par_workers};")

            # Fast chunked COPY ingestion (halfvec FP16 has ~3.3 decimal digits of mantissa precision; %.4f is lossless for FP16)
            batch_sz = 50_000
            n = len(ids)
            for start in range(0, n, batch_sz):
                end = min(n, start + batch_sz)
                v_buf = io.BytesIO()
                np.savetxt(v_buf, vecs[start:end], fmt="%.4f", delimiter=",", newline="\n")
                v_lines = v_buf.getvalue().decode("ascii").splitlines()
                out_lines = [
                    f"{int(ids[start + idx])}\t{int(tv_s[start + idx])}\t{int(tv_e[start + idx])}\t"
                    f"{int(tk_s[start + idx])}\t{int(tk_e[start + idx])}\t[{v_lines[idx]}]\n"
                    for idx in range(end - start)
                ]
                payload = "".join(out_lines)
                with cur.copy(f"COPY {self.table} (mem_id, tv_s, tv_e, tk_s, tk_e, embedding) FROM STDIN") as copy:
                    copy.write(payload)

            cur.execute(
                f"CREATE INDEX {self.table}_bitemp_btree ON {self.table} (tk_s, tv_s) INCLUDE (tk_e, tv_e, mem_id);"
            )
            cur.execute(
                f"CREATE INDEX {self.table}_vec_hnsw ON {self.table} "
                f"USING hnsw (embedding halfvec_l2_ops) WITH (m = {m}, ef_construction = {ef_construction});"
            )
            # Populate visibility map, set tuple hint bits, flush dirty ring buffers, and warm pages!
            cur.execute(f"VACUUM (ANALYZE) {self.table};")
            cur.execute("CHECKPOINT;")
            cur.execute(f"SELECT count(*) FROM {self.table} WHERE tk_s >= 0 AND tv_s >= 0;")
            cur.fetchone()
        return time.perf_counter() - t0

    def _apply_session_gucs(self, cur: "psycopg.Cursor", ef_search: int, mode: str) -> None:
        cur.execute("SET work_mem = '256MB';")
        cur.execute("SET random_page_cost = 1.1;")
        cur.execute("SET max_parallel_workers_per_gather = 0;")  # Strict 1-thread apples-to-apples comparison
        cur.execute("SET jit = off;")
        if mode == "hnsw_iterative":
            cur.execute("SET enable_seqscan = off;")
            cur.execute("SET enable_bitmapscan = off;")
            cur.execute("SET enable_sort = off;")
            cur.execute("SET enable_indexscan = on;")
            cur.execute("SET hnsw.iterative_scan = relaxed_order;")
            cur.execute(f"SET hnsw.ef_search = {int(ef_search)};")
            cur.execute(f"SET hnsw.max_scan_tuples = {max(25000, int(ef_search) * 100)};")
        else:
            # 'tuned_auto': warm-cache planner picks HNSW iterative scan vs covering B-Tree + top-K sort
            cur.execute("RESET enable_seqscan;")
            cur.execute("RESET enable_bitmapscan;")
            cur.execute("RESET enable_sort;")
            cur.execute("SET enable_indexscan = on;")
            cur.execute("SET hnsw.iterative_scan = relaxed_order;")
            cur.execute(f"SET hnsw.ef_search = {int(ef_search)};")
            cur.execute(f"SET hnsw.max_scan_tuples = {max(25000, int(ef_search) * 100)};")

    def query_batch(
        self,
        q_vecs: np.ndarray,
        tv_qs: np.ndarray,
        tk_qs: np.ndarray,
        k: int = 10,
        ef_search: int = 64,
        mode: str = "tuned_auto",
    ) -> Tuple[List[List[int]], np.ndarray]:
        all_ids: List[List[int]] = []
        latencies_us = np.zeros(len(q_vecs), dtype=np.float64)
        if len(q_vecs) == 0:
            return all_ids, latencies_us

        sql = (
            f"SELECT mem_id FROM {self.table} "
            f"WHERE tv_s <= %s AND tv_e > %s AND tk_s <= %s AND tk_e > %s "
            f"ORDER BY embedding <-> %s::halfvec({self.dim}) LIMIT %s;"
        )
        q_strs = [
            "[" + ",".join(f"{x:.4f}" for x in row) + "]"
            for row in q_vecs
        ]

        with self.conn.cursor() as cur:
            self._apply_session_gucs(cur, ef_search, mode)
            # Untimed warmup to prepare server-side plan and pull HNSW upper layers / B-Tree roots into L3/shared_buffers
            n_warm = min(6, len(q_vecs))
            for w in range(n_warm):
                cur.execute(sql, (int(tv_qs[w]), int(tv_qs[w]), int(tk_qs[w]), int(tk_qs[w]), q_strs[w], int(k)), prepare=True)
                cur.fetchall()

            for i in range(len(q_vecs)):
                tv = int(tv_qs[i])
                tk = int(tk_qs[i])
                t0 = time.perf_counter()
                cur.execute(sql, (tv, tv, tk, tk, q_strs[i], int(k)), prepare=True)
                rows = cur.fetchall()
                latencies_us[i] = (time.perf_counter() - t0) * 1e6
                all_ids.append([int(r[0]) for r in rows])

        return all_ids, latencies_us

    def total_relation_bytes(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_total_relation_size(%s);", (self.table,))
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def explain_sample(
        self, q_vec: np.ndarray, tv_q: int, tk_q: int, k: int = 10, ef_search: int = 120, mode: str = "tuned_auto"
    ) -> str:
        with self.conn.cursor() as cur:
            self._apply_session_gucs(cur, ef_search, mode)
            q_str = "[" + ",".join(f"{x:.4f}" for x in q_vec) + "]"
            sql = (
                f"EXPLAIN (ANALYZE, BUFFERS) SELECT mem_id FROM {self.table} "
                f"WHERE tv_s <= %s AND tv_e > %s AND tk_s <= %s AND tk_e > %s "
                f"ORDER BY embedding <-> %s::halfvec({self.dim}) LIMIT %s;"
            )
            cur.execute(sql, (int(tv_q), int(tv_q), int(tk_q), int(tk_q), q_str, int(k)))
            return "\n".join(str(r[0]) for r in cur.fetchall())

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
