"""Deterministic grammar/slot extractor for quantitative financial covenants & restatement anchors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pit_mem_core as pm

# Comparator op codes matching C++ CompOp enum
OP_LT = 0
OP_LE = 1
OP_GT = 2
OP_GE = 3
OP_EQ = 4
OP_NE = 5
OP_BETWEEN = 6

FLAG_NEGATED = 0x01
FLAG_CROSS_DEFAULT = 0x02
FLAG_MAC_CLAUSE = 0x04
FLAG_CURE_RIGHT = 0x08

EPOCH_BASE = int(datetime(1990, 1, 1, tzinfo=timezone.utc).timestamp())


def date_to_tick(dt_str: Optional[str]) -> int:
    if not dt_str:
        return 0
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(dt_str.strip(), fmt).replace(tzinfo=timezone.utc)
            return max(0, int(dt.timestamp()) - EPOCH_BASE)
        except ValueError:
            continue
    return 0


METRIC_PATTERNS = [
    ("Net_Debt_to_EBITDA", re.compile(r"net\s+(?:debt|leverage)\s*(?:/|to)\s*(?:adjusted\s+)?ebitda", re.I)),
    ("Total_Leverage_Ratio", re.compile(r"(?:consolidated\s+|total\s+)?leverage\s+ratio", re.I)),
    ("Interest_Coverage_Ratio", re.compile(r"interest\s+coverage\s+ratio", re.I)),
    ("Fixed_Charge_Coverage", re.compile(r"fixed\s+charge\s+coverage(?:\s+ratio)?", re.I)),
    ("Minimum_Liquidity", re.compile(r"minimum\s+(?:unrestricted\s+)?liquidity", re.I)),
    ("Unrestricted_Cash", re.compile(r"(?:unrestricted\s+)?cash(?:\s+and\s+cash\s+equivalents)?", re.I)),
    ("CapEx_Limit", re.compile(r"capital\s+expenditures?(?:\s+limit|\s+cap)?", re.I)),
    ("Consolidated_Net_Worth", re.compile(r"(?:consolidated\s+|tangible\s+)?net\s+worth", re.I)),
    ("Debt_to_Capitalization", re.compile(r"debt\s*(?:/|to)\s*capitalization", re.I)),
    ("Operating_Income", re.compile(r"(?:gaap\s+|consolidated\s+)?operating\s+(?:income|loss)", re.I)),
    ("Net_Income", re.compile(r"(?:gaap\s+|consolidated\s+)?net[\s_]+(?:income|loss)", re.I)),
    ("Total_Assets", re.compile(r"(?:consolidated\s+|total\s+)?assets", re.I)),
    ("Total_Revenue", re.compile(r"(?:total\s+|consolidated\s+)?revenues?", re.I)),
    ("Diluted_EPS", re.compile(r"(?:diluted\s+)?earnings\s+per\s+share|\beps\b", re.I)),
]

COMPARATOR_PATTERNS = [
    (OP_LE, True, re.compile(r"shall\s+not\s+exceed|must\s+not\s+exceed|not\s+to\s+exceed|at\s+most|maximum\s+of|<=", re.I)),
    (OP_GE, False, re.compile(r"shall\s+not\s+be\s+less\s+than|is\s+at\s+least|at\s+least|minimum\s+of|no\s+less\s+than|>=", re.I)),
    (OP_GT, False, re.compile(r"exceeds?|greater\s+than|above|>", re.I)),
    (OP_LT, True, re.compile(r"falls?\s+below|less\s+than|<", re.I)),
    (OP_EQ, False, re.compile(r"restated\s+to|revised\s+to|reported\s+at|equals?|=\s*", re.I)),
]

VALUE_PATTERN = re.compile(
    r"(?P<curr>\$)?\s*(?P<num>-?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>x|times|%|percent|bps|million|billion|mn|bn|m|b)?\b",
    re.I,
)

MATURITY_PATTERN = re.compile(
    r"(?:matur(?:es|ing|ity)|due|prior\s+to|through|until)\s+(?:on\s+)?(\d{4}-\d{2}-\d{2})",
    re.I,
)


def extract_risk_anchors(
    mem_id: int,
    entity: str,
    text: str,
    tv_s: int,
    tv_e: int,
    tk_s: int,
    tk_e: int,
) -> List[pm.RiskAnchorInput]:
    """Deterministically extract exact quantitative covenant & restatement anchors without LLM loss."""
    anchors: List[pm.RiskAnchorInput] = []
    # Also detect document-level maturity date fallback so multi-sentence facility clauses never lose maturity_tick
    doc_mat_tick = 0
    doc_mm = MATURITY_PATTERN.search(text)
    if doc_mm:
        doc_mat_tick = date_to_tick(doc_mm.group(1))

    # Keep semicolon-linked legal remedy clauses attached to their covenant threshold sentence
    sentences = re.split(r"(?<=\.)\s+", text)

    for sent in sentences:
        metric_spans = []
        for m_name, pat in METRIC_PATTERNS:
            for m_obj in pat.finditer(sent):
                metric_spans.append((m_obj.start(), m_obj.end(), m_name))
        if not metric_spans:
            continue
        metric_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        # Deduplicate overlapping spans (keep longest starting earliest)
        dedup_spans = []
        last_end = -1
        for st_p, en_p, mn in metric_spans:
            if st_p >= last_end:
                dedup_spans.append((st_p, en_p, mn))
                last_end = en_p

        for idx_sp, (metric_start_pos, metric_end_pos, matched_metric) in enumerate(dedup_spans):
            prev_metric_end = dedup_spans[idx_sp - 1][1] if idx_sp > 0 else 0
            next_metric_start = dedup_spans[idx_sp + 1][0] if idx_sp + 1 < len(dedup_spans) else len(sent)
            # Restrict comparator & value binding strictly to the current semicolon/metric clause span!
            semi_left = sent.rfind(";", prev_metric_end, metric_start_pos)
            local_start = (semi_left + 1) if semi_left != -1 else prev_metric_end
            semi_right = sent.find(";", metric_end_pos, next_metric_start)
            local_end = semi_right if semi_right != -1 else next_metric_start

            op_code = OP_EQ
            neg_flag = False
            val_search_start = metric_end_pos
            best_c = None
            for cand_op, is_neg, c_pat in COMPARATOR_PATTERNS:
                c_obj = c_pat.search(sent, pos=metric_end_pos, endpos=local_end)
                if c_obj is not None:
                    if best_c is None or c_obj.start() < best_c[0].start():
                        best_c = (c_obj, cand_op, is_neg)
            if best_c is None:
                for cand_op, is_neg, c_pat in COMPARATOR_PATTERNS:
                    c_obj = c_pat.search(sent, pos=local_start, endpos=metric_start_pos)
                    if c_obj is not None:
                        if best_c is None or c_obj.start() > best_c[0].start():
                            best_c = (c_obj, cand_op, is_neg)
            if best_c is not None:
                c_obj, op_code, neg_flag = best_c
                if c_obj.start() >= metric_end_pos and c_obj.end() <= local_end:
                    val_search_start = c_obj.end()

            # Find quantitative threshold / value inside the local clause (never cross into adjacent metric clauses!)
            val_matches = list(VALUE_PATTERN.finditer(sent, pos=val_search_start, endpos=local_end))
            if not val_matches:
                val_matches = list(VALUE_PATTERN.finditer(sent, pos=local_start, endpos=local_end))
            chosen = None
            for vm in val_matches:
                raw_num = vm.group("num").replace(",", "")
                try:
                    val = float(raw_num)
                except ValueError:
                    continue
                # Skip standalone 4-digit calendar years like 2025 unless accompanied by currency/unit
                if 1990 <= val <= 2040 and not vm.group("curr") and not vm.group("unit"):
                    continue
                chosen = (val, (vm.group("curr") or ""), (vm.group("unit") or "").lower())
                break

            if chosen is None:
                continue

            val, curr, unit_raw = chosen
            if unit_raw in ("x", "times"):
                unit = "x"
            elif unit_raw in ("%", "percent"):
                unit = "%"
            elif unit_raw == "bps":
                unit = "bps"
            elif unit_raw in ("billion", "bn", "b"):
                unit = "USD_B" if curr else "B"
            elif unit_raw in ("million", "mn", "m"):
                unit = "USD_M" if curr else "M"
            elif curr == "$":
                unit = "USD"
            else:
                unit = "RAW"

            flags = 0
            if neg_flag or re.search(r"\bprohibit|\brestrict|\bnegative\s+covenant", sent, re.I):
                flags |= FLAG_NEGATED
            if re.search(r"cross[- ]default|acceleration|event\s+of\s+default", sent, re.I):
                flags |= FLAG_CROSS_DEFAULT
            if re.search(r"material\s+adverse\s+(?:change|effect)|\bmac\b", sent, re.I):
                flags |= FLAG_MAC_CLAUSE
            if re.search(r"equity\s+cure|cure\s+period|grace\s+period", sent, re.I):
                flags |= FLAG_CURE_RIGHT

            if flags & FLAG_CROSS_DEFAULT:
                consequence = "CrossDefault_Acceleration"
            elif re.search(r"dividend|repurchase|restricted\s+payment", sent, re.I):
                consequence = "DividendStopper"
            elif re.search(r"item\s*4\.02|non-reliance|restat", sent, re.I):
                consequence = "Item402_Restatement"
            else:
                consequence = "CovenantConstraint"

            mat_tick = doc_mat_tick
            mm = MATURITY_PATTERN.search(sent)
            if mm:
                mat_tick = date_to_tick(mm.group(1))

            anc = pm.RiskAnchorInput()
            anc.source_mem_id = int(mem_id)
            anc.entity = entity
            anc.metric = matched_metric
            anc.unit = unit
            anc.consequence = consequence
            anc.op = int(op_code)
            anc.flags = int(flags)
            anc.threshold_val = float(val)
            anc.secondary_val = 0.0
            anc.maturity_tick = int(mat_tick)
            anc.tv_s = int(tv_s)
            anc.tv_e = int(tv_e)
            anc.tk_s = int(tk_s)
            anc.tk_e = int(tk_e)
            anchors.append(anc)

    return anchors


def evaluate_covenant_breach(op_code: int, observed_val: float, threshold_val: float) -> bool:
    """Evaluate exact legal violation of a covenant comparator (`<`, `<=`, `>`, `>=`, `=`, `!=`)."""
    if op_code == OP_LT:
        return observed_val >= threshold_val
    if op_code == OP_LE:
        return observed_val > threshold_val
    if op_code == OP_GT:
        return observed_val <= threshold_val
    if op_code == OP_GE:
        return observed_val < threshold_val
    if op_code == OP_NE:
        return abs(observed_val - threshold_val) <= 1e-9
    return abs(observed_val - threshold_val) > 1e-4
