"""SEC EDGAR Point-in-Time XBRL & Form 8-K Item 4.02 Restatement Crawler (Zero API Key Required)."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

from pit_mem.covenant_parser import date_to_tick
import pit_mem_core as pm

SEC_HEADERS = {
    "User-Agent": "PiTMemBenchResearch academic-vldb@pitmem-research.org",
    "Accept-Encoding": "gzip, deflate",
}

CORE_XBRL_TAGS = [
    ("Revenues", "Total_Revenue", "USD"),
    ("RevenueFromContractWithCustomerExcludingAssessedTax", "Total_Revenue", "USD"),
    ("SalesRevenueNet", "Total_Revenue", "USD"),
    ("NetIncomeLoss", "Net_Income", "USD"),
    ("OperatingIncomeLoss", "Operating_Income", "USD"),
    ("LongTermDebtNoncurrent", "Long_Term_Debt", "USD"),
    ("LongTermDebt", "Long_Term_Debt", "USD"),
    ("Assets", "Total_Assets", "USD"),
    ("StockholdersEquity", "Consolidated_Net_Worth", "USD"),
    ("CashAndCashEquivalentsAtCarryingValue", "Unrestricted_Cash", "USD"),
    ("EarningsPerShareDiluted", "Diluted_EPS", "USD/share"),
]

# High-coverage S&P / Nasdaq / Restatement-heavy issuers
SEED_TICKERS_CIKS = [
    ("AAPL", 320193), ("MSFT", 789019), ("NVDA", 1045810), ("AMZN", 1018724),
    ("GOOGL", 1652044), ("META", 1326801), ("TSLA", 1318605), ("JPM", 19617),
    ("BAC", 70858), ("XOM", 34088), ("BA", 12927), ("INTC", 50863),
    ("SMCI", 1375365), ("PLUG", 1093691), ("WBA", 1618921), ("PARA", 813828),
    ("CVS", 64803), ("FDX", 1048911), ("DIS", 1744489), ("NFLX", 1065280),
    ("AMD", 2488), ("CRM", 1108524), ("ORCL", 1341439), ("CSCO", 858877),
    ("PFE", 78003), ("MRK", 310158), ("UNH", 731766), ("CAT", 18230),
    ("GE", 40545), ("DE", 315189), ("LMT", 936468), ("GS", 886982),
    ("MS", 895421), ("C", 831001), ("TGT", 27419), ("COST", 909832),
]


def discover_item402_ciks(
    start_date: str = "2020-01-01",
    end_date: str = "2026-09-30",
    max_hits: int = 120,
) -> List[Tuple[str, int, str, str]]:
    """Query SEC EFTS full-text index for official Form 8-K Item 4.02 Non-Reliance filings."""
    found: List[Tuple[str, int, str, str]] = []
    seen_ciks: Set[int] = set()

    for win_start, win_end in [("2025-01-01", end_date), (start_date, "2024-12-31")]:
        for from_offset in (0, 100):
            if len(found) >= max_hits:
                break
            url = (
                "https://efts.sec.gov/LATEST/search-index"
                f"?q=%22Non-Reliance+on+Previously+Issued%22&forms=8-K"
                f"&dateRange=custom&startdt={win_start}&enddt={win_end}&from={from_offset}&size=100"
            )
            try:
                resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
                if resp.status_code != 200:
                    break
                hits = resp.json().get("hits", {}).get("hits", [])
                if not hits:
                    break
                for h in hits:
                    src = h.get("_source", {})
                    items = src.get("items", []) or []
                    if not any("4.02" in str(it) for it in items):
                        continue
                    ciks = src.get("ciks", []) or []
                    dnames = src.get("display_names", []) or []
                    fdate = src.get("file_date", "")
                    form = src.get("form", "8-K")
                    if not ciks:
                        continue
                    cik_int = int(str(ciks[0]).lstrip("0") or "0")
                    if cik_int <= 0 or cik_int in seen_ciks:
                        continue
                    seen_ciks.add(cik_int)
                    label = dnames[0] if dnames else f"CIK_{cik_int:010d}"
                    m_sym = re.search(r"\(([A-Z0-9.\-]{1,8})\)", label)
                    sym = m_sym.group(1) if m_sym else f"CIK{cik_int}"
                    found.append((sym, cik_int, fdate, form))
                    if len(found) >= max_hits:
                        break
            except Exception as exc:
                print(f"[SEC EFTS warn] {exc}")
                break
            time.sleep(0.16)
    return found


def _format_val(val: float, unit_kind: str) -> str:
    if unit_kind == "USD/share":
        return f"${val:.2f}"
    abs_v = abs(val)
    if abs_v >= 1e9:
        return f"${val / 1e9:.3f} billion"
    if abs_v >= 1e6:
        return f"${val / 1e6:.2f} million"
    return f"${val:,.0f}"


def extract_company_pit_facts(
    symbol: str,
    cik: int,
    item402_date: Optional[str],
    base_mem_id: int,
) -> Tuple[List[dict], int]:
    """Fetch SEC XBRL companyfacts JSON and extract authentic Point-in-Time EVOLVE & RESTATE chains."""
    records: List[dict] = []
    cur_id = base_mem_id

    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    resp = requests.get(facts_url, headers=SEC_HEADERS, timeout=20)
    if resp.status_code != 200:
        return records, cur_id

    payload = resp.json()
    entity_name = payload.get("entityName", symbol)
    us_gaap = payload.get("facts", {}).get("us-gaap", {})

    # Record official Form 8-K Item 4.02 event if present
    if item402_date:
        tk_402 = date_to_tick(item402_date)
        tv_402_s = max(0, tk_402 - 365 * 86400)
        records.append({
            "mem_id": cur_id,
            "source": "SEC_8K_ITEM_402",
            "entity": symbol,
            "entity_name": entity_name,
            "cik": cik,
            "metric": "Accounting_Reliability",
            "obs_period": item402_date[:7],
            "value": 0.0,
            "prev_value": 1.0,
            "op_kind": "RESTATE_NOTICE",
            "tv_s": tv_402_s,
            "tv_e": tk_402,
            "tk_s": tk_402,
            "tk_e": int(pm.INF_TICK),
            "valid_start_iso": item402_date,
            "tx_start_iso": item402_date,
            "is_post_cutoff_2025": item402_date >= "2025-01-01",
            "replaces_mem_id": 0,
            "text": (
                f"[SEC FORM 8-K ITEM 4.02 NON-RELIANCE] {entity_name} ({symbol}, CIK {cik:010d}) filed Form 8-K "
                f"Item 4.02 on {item402_date} declaring that previously issued financial statements should no "
                f"longer be relied upon due to accounting misstatements requiring restatement."
            ),
        })
        cur_id += 1

    seen_canonical_Period: Set[Tuple[str, str, str, str]] = set()
    for tag_name, canonical_metric, unit_kind in CORE_XBRL_TAGS:
        tag_obj = us_gaap.get(tag_name)
        if not tag_obj:
            continue
        units_map = tag_obj.get("units", {})
        entries: List[dict] = []
        for u_key, u_list in units_map.items():
            if unit_kind == "USD" and "USD" not in u_key:
                continue
            entries.extend(u_list)

        if not entries:
            continue

        # Group observations by exact valid period (start, end) for durational metrics or (_, end) for point-in-time balance sheet
        by_period: Dict[Tuple[str, str], List[dict]] = {}
        for e in entries:
            filed = e.get("filed", "")
            end_dt = e.get("end", "")
            start_dt = e.get("start", end_dt)
            form = e.get("form", "")
            if not filed or not end_dt or end_dt < "2017-01-01":
                continue
            if form not in ("10-Q", "10-K", "10-Q/A", "10-K/A", "8-K"):
                continue
            # Avoid mixing 3-month and 9-month windows: keep exact (start_dt, end_dt) pairs
            key = (start_dt, end_dt)
            if (canonical_metric, start_dt, end_dt, filed) in seen_canonical_Period:
                continue
            seen_canonical_Period.add((canonical_metric, start_dt, end_dt, filed))
            by_period.setdefault(key, []).append(e)

        # Also track chronological progression across consecutive periods for EVOLVE
        prev_period_rec_idx: Optional[int] = None

        for (p_start, p_end), p_list in sorted(by_period.items()):
            # Sort filings for this SAME valid period chronologically by transaction date `filed`
            p_list.sort(key=lambda x: (x.get("filed", ""), 1 if "/A" in x.get("form", "") else 0))
            tv_s = date_to_tick(p_start)
            tv_e = max(tv_s + 86400, date_to_tick(p_end) + 86400)

            active_idx_for_period: Optional[int] = None
            active_val_for_period: Optional[float] = None

            for filing in p_list:
                try:
                    val = float(filing["val"])
                except (KeyError, TypeError, ValueError):
                    continue
                filed_dt = filing["filed"]
                form = filing.get("form", "10-Q")
                tk_s = date_to_tick(filed_dt)

                if active_idx_for_period is None:
                    op_kind = "EVOLVE" if prev_period_rec_idx is not None else "INSERT"
                    val_fmt = _format_val(val, unit_kind)
                    text = (
                        f"[SEC {form} FILING] {entity_name} ({symbol}) reported {canonical_metric.replace('_', ' ')} "
                        f"at {val_fmt} for fiscal period {p_start} to {p_end} (filed {filed_dt})."
                    )
                    active_idx_for_period = len(records)
                    active_val_for_period = val
                    records.append({
                        "mem_id": cur_id,
                        "source": "SEC_EDGAR_XBRL",
                        "entity": symbol,
                        "entity_name": entity_name,
                        "cik": cik,
                        "metric": canonical_metric,
                        "obs_period": f"{p_start}..{p_end}",
                        "value": val,
                        "prev_value": None,
                        "op_kind": op_kind,
                        "tv_s": tv_s,
                        "tv_e": tv_e,
                        "tk_s": tk_s,
                        "tk_e": int(pm.INF_TICK),
                        "valid_start_iso": p_start,
                        "tx_start_iso": filed_dt,
                        "is_post_cutoff_2025": filed_dt >= "2025-01-01",
                        "replaces_mem_id": records[prev_period_rec_idx]["mem_id"] if prev_period_rec_idx is not None else 0,
                        "text": text,
                    })
                    prev_period_rec_idx = active_idx_for_period
                    cur_id += 1
                else:
                    # Check if later filing CHANGED the numerical value for the EXACT SAME historical period!
                    assert active_val_for_period is not None
                    rel_diff = abs(val - active_val_for_period) / max(abs(active_val_for_period), 1e-6)
                    if rel_diff >= 0.002 and tk_s > records[active_idx_for_period]["tk_s"]:
                        records[active_idx_for_period]["tk_e"] = tk_s
                        old_mid = records[active_idx_for_period]["mem_id"]
                        old_fmt = _format_val(active_val_for_period, unit_kind)
                        new_fmt = _format_val(val, unit_kind)
                        pct = ((val - active_val_for_period) / max(abs(active_val_for_period), 1e-6)) * 100.0
                        text = (
                            f"[SEC {form} FINANCIAL RESTATEMENT] {entity_name} ({symbol}) restated historical "
                            f"{canonical_metric.replace('_', ' ')} for period {p_start} to {p_end}: revised to {new_fmt} "
                            f"(previously reported at {old_fmt}, adjustment {pct:+.2f}%) in filing on {filed_dt}."
                        )
                        new_idx = len(records)
                        records.append({
                            "mem_id": cur_id,
                            "source": "SEC_EDGAR_XBRL",
                            "entity": symbol,
                            "entity_name": entity_name,
                            "cik": cik,
                            "metric": canonical_metric,
                            "obs_period": f"{p_start}..{p_end}",
                            "value": val,
                            "prev_value": active_val_for_period,
                            "op_kind": "RESTATE",
                            "tv_s": tv_s,
                            "tv_e": tv_e,
                            "tk_s": tk_s,
                            "tk_e": int(pm.INF_TICK),
                            "valid_start_iso": p_start,
                            "tx_start_iso": filed_dt,
                            "is_post_cutoff_2025": filed_dt >= "2025-01-01",
                            "replaces_mem_id": old_mid,
                            "text": text,
                        })
                        active_idx_for_period = new_idx
                        active_val_for_period = val
                        cur_id += 1

    return records, cur_id


def build_sec_edgar_pit_dataset(
    output_parquet: str,
    max_item402_issuers: int = 75,
) -> pd.DataFrame:
    """Crawl real SEC Form 8-K Item 4.02 issuers + S&P liquid issuers and persist Point-in-Time Parquet."""
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    item402_list = discover_item402_ciks(max_hits=max_item402_issuers)
    print(f"[SEC EDGAR] Discovered {len(item402_list)} real Form 8-K Item 4.02 issuers (2020-2026).")

    targets: Dict[int, Tuple[str, Optional[str]]] = {}
    for sym, cik, fdate, _ in item402_list:
        targets[cik] = (sym, fdate)
    for sym, cik in SEED_TICKERS_CIKS:
        if cik not in targets:
            targets[cik] = (sym, None)

    all_records: List[dict] = []
    cur_id = 2_000_000
    for idx, (cik, (sym, fdate)) in enumerate(targets.items()):
        try:
            recs, cur_id = extract_company_pit_facts(sym, cik, fdate, cur_id)
            all_records.extend(recs)
        except Exception as exc:
            print(f"[SEC warn] {sym} ({cik}): {exc}")
        time.sleep(0.16) # Strictly <= 6.25 req/s per SEC fair access policy

    df = pd.DataFrame(all_records)
    df.to_parquet(output_parquet, index=False)
    n_restate = int(df["op_kind"].isin(["RESTATE", "RESTATE_NOTICE"]).sum()) if len(df) else 0
    n_post2025 = int((df["is_post_cutoff_2025"] & df["op_kind"].isin(["RESTATE", "RESTATE_NOTICE"])).sum()) if len(df) else 0
    print(
        f"[SEC EDGAR] Saved {len(df)} Point-in-Time XBRL/8-K records across {len(targets)} issuers "
        f"({n_restate} total restatement events, {n_post2025} post-2025-01-01 cutoff!) -> {output_parquet}"
    )
    return df


if __name__ == "__main__":
    build_sec_edgar_pit_dataset("data/raw_seed/sec_edgar_pit.parquet", max_item402_issuers=75)
