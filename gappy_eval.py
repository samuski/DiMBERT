#!/usr/bin/env python3
"""
Exact-group MWE scorer for DiMSUM files, split into gappy vs contiguous groups.

Complements the official dimsumeval.py (which reports link-level MWE scores and
cross-gap link P/R). This script reports exact matches of whole MWE groups:

  * A group is the set of token positions of one MWE, recovered from the
    6-tag scheme (B/I outer expression; b/i expression inside a gap; o a
    single word inside a gap; O outside). Groups of size 1 are not MWEs.
  * A group is gappy if its token set is non-contiguous.
  * A predicted group is a true positive only if a gold group has exactly the
    same token set. Matching is one-to-one by identity.

The same group definition is applied to gold and predicted files.

Usage:
  python gappy_eval.py GOLD_FILE PRED_FILE [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

Group = FrozenSet[int]


def read_tag_sentences(path: Path) -> List[List[str]]:
    """Return the MWE tag (column 5) of every token, sentence by sentence."""
    sentences: List[List[str]] = []
    current: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            current.append(cols[4] if cols[4] else "O")
    if current:
        sentences.append(current)
    return sentences


def groups_from_tags(tags: Sequence[str]) -> List[Group]:
    """Recover MWE groups (0-based token positions) from a 6-tag sequence."""
    groups: List[Group] = []
    outer: List[int] = []
    inner: List[int] = []

    def close_outer():
        nonlocal outer
        if len(outer) >= 2:
            groups.append(frozenset(outer))
        outer = []

    def close_inner():
        nonlocal inner
        if len(inner) >= 2:
            groups.append(frozenset(inner))
        inner = []

    for idx, tag in enumerate(tags):
        if tag == "B":
            close_inner()
            close_outer()
            outer = [idx]
        elif tag == "I":
            close_inner()
            outer.append(idx)
        elif tag == "b":
            close_inner()
            inner = [idx]
        elif tag == "i":
            inner.append(idx)
        elif tag == "o":
            close_inner()
        else:  # "O" or anything unexpected
            close_inner()
            close_outer()
    close_inner()
    close_outer()
    return groups


def is_gappy(group: Group) -> bool:
    return max(group) - min(group) + 1 != len(group)


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(gold_path: Path, pred_path: Path) -> Dict[str, Dict[str, float]]:
    gold = read_tag_sentences(gold_path)
    pred = read_tag_sentences(pred_path)
    if len(gold) != len(pred):
        raise ValueError(f"Sentence count mismatch: gold {len(gold)} vs pred {len(pred)}")

    counts = {k: {"tp": 0, "fp": 0, "fn": 0, "gold": 0, "pred": 0} for k in ("gappy", "contiguous", "all")}

    for s_idx, (g_tags, p_tags) in enumerate(zip(gold, pred)):
        if len(g_tags) != len(p_tags):
            raise ValueError(f"Token count mismatch in sentence {s_idx + 1}")
        g_groups: Set[Group] = set(groups_from_tags(g_tags))
        p_groups: Set[Group] = set(groups_from_tags(p_tags))

        for kind in ("gappy", "contiguous", "all"):
            if kind == "all":
                g_sel, p_sel = g_groups, p_groups
            else:
                want = kind == "gappy"
                g_sel = {g for g in g_groups if is_gappy(g) == want}
                p_sel = {g for g in p_groups if is_gappy(g) == want}
            tp = len(g_sel & p_sel)
            c = counts[kind]
            c["tp"] += tp
            c["fp"] += len(p_sel) - tp
            c["fn"] += len(g_sel) - tp
            c["gold"] += len(g_sel)
            c["pred"] += len(p_sel)

    out: Dict[str, Dict[str, float]] = {}
    for kind, c in counts.items():
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        out[kind] = {**c, "precision": round(100 * p, 2), "recall": round(100 * r, 2), "f1": round(100 * f, 2)}
    return out


def format_report(result: Dict[str, Dict[str, float]]) -> str:
    lines = ["Exact-group MWE scores (gold groups / predicted groups / TP FP FN / P R F1)"]
    for kind in ("gappy", "contiguous", "all"):
        c = result[kind]
        lines.append(
            f"{kind:>10}: gold={c['gold']} pred={c['pred']} "
            f"TP={c['tp']} FP={c['fp']} FN={c['fn']} "
            f"P={c['precision']:.2f} R={c['recall']:.2f} F1={c['f1']:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gold", type=Path)
    parser.add_argument("pred", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write the scores as JSON.")
    args = parser.parse_args()

    result = score(args.gold, args.pred)
    print(format_report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
