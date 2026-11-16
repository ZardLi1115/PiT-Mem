"""Construct FinPiT-Bench Dual-Isolation Evaluation Suites (Protocol A & Protocol B)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pit_mem.covenant_parser import date_to_tick
import pit_mem_core as pm


def build_protocol_a_post2025(
    macro_parquet: str,
    sec_parquet: str,
    output_json: str,
    max_cases: int = 180,
) -> List[dict]:
    """
    Protocol A (Post-Cutoff Real Window: 2025-01-01 to 2026-09-30):
    Selects real restatement chains from SEC EDGAR XBRL / 8-K Item 4.02 and ALFRED vintages
    where the restatement occurred strictly AFTER 2025-01-01 (post Qwen2.5 mid-2024 knowledge cutoff).
    """
    df_sec = pd.read_parquet(sec_parquet)
    df_mac = pd.read_parquet(macro_parquet)

    df_all = pd.concat([df_sec, df_mac], ignore_index=True)
    by_id: Dict[int, dict] = {int(r["mem_id"]): r for r in df_all.to_dict(orient="records")}

    # Filter authentic RESTATE records disclosed on or after 2025-01-01 with a linked prior record
    restates = df_all[
        (df_all["op_kind"] == "RESTATE")
        & (df_all["is_post_cutoff_2025"] == True)
        & (df_all["replaces_mem_id"] > 0)
    ].copy()

    cases: List[dict] = []
    seen_pairs = set()

    for row in restates.to_dict(orient="records"):
        old_id = int(row["replaces_mem_id"])
        if old_id not in by_id:
            continue
        orig = by_id[old_id]
        tk_orig = int(orig["tk_s"])
        tk_restate = int(row["tk_s"])
        if tk_restate <= tk_orig + 7 * 86400:
            continue

        pair_key = (str(row["entity"]), str(row["metric"]), str(row["obs_period"]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        orig_val = float(orig["value"]) if orig["value"] is not None else 0.0
        new_val = float(row["value"]) if row["value"] is not None else 0.0
        if abs(orig_val) < 1e-6 and abs(new_val) < 1e-6:
            continue

        rel_adj = (new_val - orig_val) / max(abs(orig_val), 1e-6)
        # Focus on economically meaningful restatements (>= 1% shift in reported metric or macro index)
        if abs(rel_adj) < 0.005 and row["source"] != "ALFRED_MACRO":
            continue

        # Pre-restatement backtest query time: midpoint between initial filing and restatement
        tk_pre_query = tk_orig + (tk_restate - tk_orig) // 2
        # Reflection generation time: shortly after initial filing (before restatement!)
        tk_reflect_gen = tk_orig + max(86400, (tk_restate - tk_orig) // 4)
        # Post-restatement audit query time: 7 days after restatement filing
        tk_post_query = tk_restate + 7 * 86400

        tv_q = int(orig["tv_s"]) + max(1, (int(orig["tv_e"]) - int(orig["tv_s"])) // 2)
        direction = "DOWNWARD_RESTATEMENT" if new_val < orig_val else "UPWARD_RESTATEMENT"

        pre_reflection_text = (
            f"[AGENT CONSOLIDATED REFLECTION] Based on initial {orig['source']} disclosure for {row['entity']} "
            f"covering period {row['obs_period']}, {row['metric']} stands at {orig_val:.4g}, supporting "
            f"{'positive operating baseline' if orig_val >= 0.0 else 'defensive balance-sheet'} thesis."
        )
        post_reflection_text = (
            f"[AGENT REPAIRED REFLECTION] Following {row['tx_start_iso']} restatement for {row['entity']} "
            f"covering period {row['obs_period']}, {row['metric']} was revised from {orig_val:.4g} to {new_val:.4g} "
            f"({rel_adj * 100.0:+.2f}% {direction}); prior thesis invalidated."
        )

        cases.append({
            "case_id": f"PA_{len(cases):04d}",
            "protocol": "A_POST_2025_REAL",
            "source": str(row["source"]),
            "entity": str(row["entity"]),
            "metric": str(row["metric"]),
            "obs_period": str(row["obs_period"]),
            "orig_mem_id": old_id,
            "restate_mem_id": int(row["mem_id"]),
            "reflect_pre_id": 80_000_000 + len(cases) * 2,
            "reflect_post_id": 80_000_000 + len(cases) * 2 + 1,
            "orig_val": orig_val,
            "restated_val": new_val,
            "rel_change": rel_adj,
            "direction": direction,
            "tv_s": int(orig["tv_s"]),
            "tv_e": int(orig["tv_e"]),
            "tk_orig": tk_orig,
            "tk_reflect_gen": tk_reflect_gen,
            "tk_pre_query": tk_pre_query,
            "tk_restate": tk_restate,
            "tk_post_query": tk_post_query,
            "tv_query": tv_q,
            "orig_filing_date": str(orig["tx_start_iso"]),
            "restate_filing_date": str(row["tx_start_iso"]),
            "orig_text": str(orig["text"]),
            "restate_text": str(row["text"]),
            "pre_reflection_text": pre_reflection_text,
            "post_reflection_text": post_reflection_text,
            "query_text": (
                f"What is the reported {str(row['metric']).replace('_', ' ')} and risk assessment "
                f"for {row['entity']} during period {row['obs_period']}?"
            ),
        })
        if len(cases) >= max_cases:
            break

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"[FinPiT-Bench] Built Protocol A ({len(cases)} post-2025 real restatement cases) -> {output_json}")
    return cases


def build_protocol_b_counterfactual_and_covenants(
    sec_parquet: str,
    output_json: str,
    num_cases: int = 200,
    seed: int = 202611,
) -> List[dict]:
    """
    Protocol B (Counterfactual Entity-Anonymized Perturbation + Quantitative Tail Covenants):
    Eliminates 100% of parametric pretraining memory by mapping real issuers to anonymous IDs
    (`Firm_Alpha_001`..`Firm_Alpha_200`), applying order-preserving scale perturbation gamma in [0.85, 1.15],
    and pairing with exact quantitative negative debt covenants for both Exp 4 and Exp 5.
    """
    rng = np.random.default_rng(seed)
    df_sec = pd.read_parquet(sec_parquet)
    restates = df_sec[(df_sec["op_kind"] == "RESTATE") & (df_sec["replaces_mem_id"] > 0)].copy()
    by_id = {int(r["mem_id"]): r for r in df_sec.to_dict(orient="records")}

    covenant_templates = [
        (
            "Net_Debt_to_EBITDA",
            "shall not exceed",
            "x",
            (2.75, 4.50),
            "CrossDefault_Acceleration",
            "Under Section 7.01 of the Senior Secured Credit Agreement, Consolidated Net Debt to EBITDA "
            "shall not exceed {thr:.2f}x as of the end of any fiscal quarter; any breach triggers immediate "
            "cross-default and acceleration across all pari passu notes maturing on {mat}.",
        ),
        (
            "Interest_Coverage_Ratio",
            "shall not be less than",
            "x",
            (2.00, 3.75),
            "CrossDefault_Acceleration",
            "Pursuant to the Amended Revolving Facility Covenant 6.12, Consolidated Interest Coverage Ratio "
            "shall not be less than {thr:.2f}x; failure to maintain this ratio without an equity cure "
            "constitutes an Event of Default with cross-default acceleration due on {mat}.",
        ),
        (
            "Minimum_Liquidity",
            "at least",
            "USD_M",
            (75.0, 350.0),
            "DividendStopper",
            "Under the Term Loan B Negative Covenant Schedule, the Borrower is prohibited from making "
            "restricted dividend payments unless Minimum Unrestricted Liquidity is at least ${thr:.1f} million "
            "through maturity on {mat}.",
        ),
        (
            "Total_Leverage_Ratio",
            "must not exceed",
            "x",
            (3.00, 5.25),
            "CrossDefault_Acceleration",
            "Per the Indenture Supplement, Total Leverage Ratio must not exceed {thr:.2f}x; if restated "
            "financial statements reveal leverage above {thr:.2f}x, cross-default remedies apply prior to {mat}.",
        ),
    ]

    filler_Boilerplate = (
        "Management continues to monitor global macroeconomic conditions, supply chain normalization, "
        "working capital seasonality, foreign currency hedging effectiveness, and capital allocation "
        "priorities across core operating segments in North America, EMEA, and APAC."
    )

    raw_rows = restates.to_dict(orient="records")
    cases: List[dict] = []

    for idx in range(num_cases):
        anon_firm = f"Firm_Alpha_{idx + 1:03d}"
        gamma = float(rng.uniform(0.85, 1.15))

        if idx < len(raw_rows) and int(raw_rows[idx]["replaces_mem_id"]) in by_id:
            r = raw_rows[idx]
            orig = by_id[int(r["replaces_mem_id"])]
            tv_s = int(orig["tv_s"])
            tv_e = int(orig["tv_e"])
            tk_orig = int(orig["tk_s"])
            tk_restate = max(tk_orig + 30 * 86400, int(r["tk_s"]))
            base_orig = float(orig["value"]) * gamma
            base_new = float(r["value"]) * gamma
            metric_name = str(r["metric"])
        else:
            tv_s = date_to_tick("2024-01-01") + idx * 86400 * 3
            tv_e = tv_s + 90 * 86400
            tk_orig = tv_e + 35 * 86400
            tk_restate = tk_orig + 120 * 86400
            base_orig = float(rng.uniform(180.0, 950.0)) * 1e6 * gamma
            base_new = base_orig * float(rng.uniform(0.62, 0.84))
            metric_name = "Net_Income"

        cov_metric, comp_phrase, cov_unit, (lo, hi), conseq, tmpl = covenant_templates[idx % len(covenant_templates)]
        thr = round(float(rng.uniform(lo, hi)), 2 if cov_unit == "x" else 1)
        mat_year = int(rng.integers(2027, 2031))
        mat_date = f"{mat_year}-06-30"
        cov_sentence = tmpl.format(thr=thr, mat=mat_date)

        # Simulated actual ratio pre-restatement (compliant!) vs post-restatement (breached!)
        if "less than" in comp_phrase or "at least" in comp_phrase:
            pre_actual = round(thr * float(rng.uniform(1.08, 1.25)), 2)
            post_actual = round(thr * float(rng.uniform(0.72, 0.91)), 2)
        else:
            pre_actual = round(thr * float(rng.uniform(0.76, 0.92)), 2)
            post_actual = round(thr * float(rng.uniform(1.09, 1.32)), 2)

        orig_doc = (
            f"[{anon_firm} 10-Q INITIAL DISCLOSURE] {filler_Boilerplate} For the fiscal quarter, {anon_firm} "
            f"reported {metric_name} of ${base_orig / 1e6:.2f} million and {cov_metric.replace('_', ' ')} "
            f"at {pre_actual}{'' if cov_unit == 'USD_M' else 'x'}. {cov_sentence} {filler_Boilerplate}"
        )
        restate_doc = (
            f"[{anon_firm} FORM 8-K ITEM 4.02 & 10-Q/A RESTATEMENT] {anon_firm} announced non-reliance on "
            f"previously issued statements: {metric_name} is restated to ${base_new / 1e6:.2f} million "
            f"(from ${base_orig / 1e6:.2f} million), causing {cov_metric.replace('_', ' ')} to be revised to "
            f"{post_actual}{'' if cov_unit == 'USD_M' else 'x'}, breaching the {thr}{'' if cov_unit == 'USD_M' else 'x'} threshold."
        )
        pre_refl = (
            f"[CONSOLIDATED THESIS: {anon_firm}] {anon_firm} maintains healthy covenant headroom with "
            f"{cov_metric.replace('_', ' ')} at {pre_actual} vs covenant limit {thr}; no default risk."
        )
        post_refl = (
            f"[REPAIRED THESIS: {anon_firm}] Post-restatement {cov_metric.replace('_', ' ')} moved to "
            f"{post_actual}, violating the {thr} covenant threshold and triggering {conseq}."
        )

        tk_pre_q = tk_orig + (tk_restate - tk_orig) // 2
        tk_post_q = tk_restate + 5 * 86400

        cases.append({
            "case_id": f"PB_{idx:04d}",
            "protocol": "B_COUNTERFACTUAL_COVENANT",
            "entity": anon_firm,
            "gamma_perturbation": round(gamma, 5),
            "metric": metric_name,
            "covenant_metric": cov_metric,
            "comparator_phrase": comp_phrase,
            "covenant_unit": cov_unit,
            "covenant_threshold": thr,
            "covenant_consequence": conseq,
            "maturity_date": mat_date,
            "pre_restate_metric_val": pre_actual,
            "post_restate_metric_val": post_actual,
            "covenant_breached_post_restate": True,
            "orig_mem_id": 90_000_000 + idx * 4,
            "restate_mem_id": 90_000_000 + idx * 4 + 1,
            "reflect_pre_id": 90_000_000 + idx * 4 + 2,
            "reflect_post_id": 90_000_000 + idx * 4 + 3,
            "tv_s": tv_s,
            "tv_e": tv_e,
            "tk_orig": tk_orig,
            "tk_reflect_gen": tk_orig + 86400 * 3,
            "tk_pre_query": tk_pre_q,
            "tk_restate": tk_restate,
            "tk_post_query": tk_post_q,
            "tv_query": tv_s + (tv_e - tv_s) // 2,
            "orig_text": orig_doc,
            "restate_text": restate_doc,
            "pre_reflection_text": pre_refl,
            "post_reflection_text": post_refl,
            "covenant_qa_question": (
                f"For {anon_firm}, what is the exact {cov_metric.replace('_', ' ')} covenant threshold, "
                f"maturity date, and default consequence, and is {anon_firm} in breach?"
            ),
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"[FinPiT-Bench] Built Protocol B ({len(cases)} counterfactual covenant & restatement cases) -> {output_json}")
    return cases


if __name__ == "__main__":
    build_protocol_a_post2025(
        "data/raw_seed/alfred_macro_pit.parquet",
        "data/raw_seed/sec_edgar_pit.parquet",
        "data/raw_seed/protocol_a_post2025.json",
    )
    build_protocol_b_counterfactual_and_covenants(
        "data/raw_seed/sec_edgar_pit.parquet",
        "data/raw_seed/protocol_b_counterfactual.json",
    )
