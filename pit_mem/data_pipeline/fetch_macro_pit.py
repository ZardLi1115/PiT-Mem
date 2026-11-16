"""Keyless St. Louis Fed ALFRED multi-vintage macroeconomic revision crawler & bi-temporal builder."""

from __future__ import annotations

import csv
import io
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd
import requests

from pit_mem.covenant_parser import date_to_tick
import pit_mem_core as pm

ALFRED_SERIES = [
    ("GDPC1", "Real Gross Domestic Product", "Billions of Chained 2017 Dollars", "Quarterly"),
    ("PAYEMS", "All Employees, Total Nonfarm Payrolls", "Thousands of Persons", "Monthly"),
    ("CPIAUCSL", "Consumer Price Index for All Urban Consumers", "Index 1982-1984=100", "Monthly"),
    ("RSAFS", "Advance Retail Sales: Retail Trade and Food Services", "Millions of Dollars", "Monthly"),
    ("INDPRO", "Industrial Production: Total Index", "Index 2017=100", "Monthly"),
    ("PCEPI", "Personal Consumption Expenditures: Chain-type Price Index", "Index 2017=100", "Monthly"),
    ("UNRATE", "Unemployment Rate", "Percent", "Monthly"),
]

# Vintage snapshots spanning 2019-Q1 through 2026-Q3 (covering both pre-cutoff and post-cutoff windows!)
DEFAULT_VINTAGE_DATES = [
    "2019-06-01", "2020-01-15", "2020-07-15", "2021-01-15", "2021-07-15",
    "2022-01-15", "2022-07-15", "2023-01-15", "2023-07-15", "2024-01-15",
    "2024-07-15", "2025-02-15", "2025-06-15", "2025-10-15", "2026-02-15",
    "2026-06-15", "2026-09-15",
]


def _period_end_date(obs_date: str, freq: str) -> str:
    dt = datetime.strptime(obs_date, "%Y-%m-%d")
    if freq == "Quarterly":
        # Approximate 90-day quarter window
        month = dt.month + 3
        year = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        return f"{year:04d}-{month:02d}-01"
    else:
        month = dt.month + 1
        year = dt.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        return f"{year:04d}-{month:02d}-01"


def fetch_alfred_vintage(series_id: str, vintage_date: str, timeout: int = 15) -> Dict[str, float]:
    url = f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={series_id}&vintage_date={vintage_date}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    out: Dict[str, float] = {}
    reader = csv.reader(io.StringIO(resp.text))
    header = next(reader, None)
    if not header:
        return out
    for row in reader:
        if len(row) < 2:
            continue
        obs_dt, val_str = row[0].strip(), row[1].strip()
        if obs_dt < "2016-01-01" or val_str in ("", ".", "ND"):
            continue
        try:
            out[obs_dt] = float(val_str)
        except ValueError:
            continue
    return out


def build_macro_pit_records(
    output_parquet: str,
    vintages: List[str] = DEFAULT_VINTAGE_DATES,
) -> pd.DataFrame:
    """Fetch real ALFRED multi-vintage macro series and build bi-temporal EVOLVE & RESTATE chains."""
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    records: List[dict] = []
    rec_id = 1_000_000

    for series_id, title, units, freq in ALFRED_SERIES:
        # Track active record index in `records` for each observation_date
        active_by_obs: Dict[str, Tuple[int, float]] = {}
        for v_date in vintages:
            try:
                snap = fetch_alfred_vintage(series_id, v_date)
            except Exception as exc:
                print(f"[ALFRED warn] {series_id} @ {v_date}: {exc}")
                continue

            tk_now = date_to_tick(v_date)
            for obs_dt, val in sorted(snap.items()):
                tv_s = date_to_tick(obs_dt)
                tv_e = date_to_tick(_period_end_date(obs_dt, freq))
                if obs_dt not in active_by_obs:
                    idx = len(records)
                    text = (
                        f"[MACRO INITIAL RELEASE] US {title} ({series_id}) for period starting {obs_dt} "
                        f"reported at {val:.3f} {units} as of ALFRED vintage {v_date}."
                    )
                    records.append({
                        "mem_id": rec_id,
                        "source": "ALFRED_MACRO",
                        "entity": f"US_MACRO_{series_id}",
                        "metric": series_id,
                        "obs_period": obs_dt,
                        "value": val,
                        "prev_value": None,
                        "op_kind": "INSERT",
                        "tv_s": tv_s,
                        "tv_e": tv_e,
                        "tk_s": tk_now,
                        "tk_e": int(pm.INF_TICK),
                        "valid_start_iso": obs_dt,
                        "tx_start_iso": v_date,
                        "is_post_cutoff_2025": v_date >= "2025-01-01",
                        "replaces_mem_id": 0,
                        "text": text,
                    })
                    active_by_obs[obs_dt] = (idx, val)
                    rec_id += 1
                else:
                    prev_idx, prev_val = active_by_obs[obs_dt]
                    if abs(val - prev_val) > 1e-4:
                        # Authentic historical macro revision (RESTATE on the same observation period!)
                        records[prev_idx]["tk_e"] = tk_now
                        old_mid = records[prev_idx]["mem_id"]
                        delta_pct = ((val - prev_val) / max(abs(prev_val), 1e-6)) * 100.0
                        idx = len(records)
                        text = (
                            f"[MACRO BENCHMARK RESTATEMENT] US {title} ({series_id}) for historical period {obs_dt} "
                            f"restated to {val:.3f} {units} (revised from previously reported {prev_val:.3f}, "
                            f"delta {delta_pct:+.2f}%) in ALFRED vintage {v_date}."
                        )
                        records.append({
                            "mem_id": rec_id,
                            "source": "ALFRED_MACRO",
                            "entity": f"US_MACRO_{series_id}",
                            "metric": series_id,
                            "obs_period": obs_dt,
                            "value": val,
                            "prev_value": prev_val,
                            "op_kind": "RESTATE",
                            "tv_s": tv_s,
                            "tv_e": tv_e,
                            "tk_s": tk_now,
                            "tk_e": int(pm.INF_TICK),
                            "valid_start_iso": obs_dt,
                            "tx_start_iso": v_date,
                            "is_post_cutoff_2025": v_date >= "2025-01-01",
                            "replaces_mem_id": old_mid,
                            "text": text,
                        })
                        active_by_obs[obs_dt] = (idx, val)
                        rec_id += 1
            time.sleep(0.08)

    df = pd.DataFrame(records)
    df.to_parquet(output_parquet, index=False)
    print(
        f"[ALFRED] Saved {len(df)} bi-temporal macro records "
        f"({int((df['op_kind'] == 'RESTATE').sum())} authentic revisions) -> {output_parquet}"
    )
    return df


if __name__ == "__main__":
    build_macro_pit_records("data/raw_seed/alfred_macro_pit.parquet")
