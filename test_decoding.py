#!/usr/bin/env python3
"""
Offline checks for MWE decoding and scoring (no trained model needed; CPU, ~1 min).

Run from the repo root (needs dimsum-data/):
  python test_decoding.py

Checks:
  1. clean_mwe_tags leaves every gold tag sequence unchanged.
  2. Every gold tag bigram / start / end tag is allowed by the B+c constraint table.
  3. G decoder round trip: gold parent links -> decode_parent_links -> tags equal the
     gold tags for every train and test sentence, with no links removed.
  4. B+c Viterbi: with no constraints it matches torchcrf's decode; with constraints it
     returns the unconstrained path whenever that path is already valid, and only
     valid paths otherwise.
  5. gappy_eval on gold vs gold: 837 groups, 36 gappy (official scorer counts).
"""

import sys
from pathlib import Path

import torch

import dimsum_unified as du
import gappy_eval as ge

DATA = Path("dimsum-data")


def is_valid(tags):
    return (tags[0] in du.MWE_ALLOWED_START and tags[-1] in du.MWE_ALLOWED_END
            and all(b in du.MWE_ALLOWED_NEXT[a] for a, b in zip(tags, tags[1:])))


def gold_logprobs(sentence):
    n = len(sentence)
    lp = torch.full((n, n), -50.0)
    for j, (_, _, _, parent) in enumerate(sentence):
        p = parent - 1
        lp[j, p if p >= 0 else j] = 0.0
    return lp


def check_gold(split):
    data = du.parse_dimsum_file(DATA / split)
    for k, s in enumerate(data):
        gold = [x[1] for x in s]
        assert du.clean_mwe_tags(gold) == gold, f"{split} sentence {k + 1}: clean_mwe_tags changed gold"
        assert is_valid(gold), f"{split} sentence {k + 1}: gold tags not allowed by constraint table"
        tags, log = du.decode_parent_links(gold_logprobs(s), torch.arange(len(s)))
        assert tags == gold, f"{split} sentence {k + 1}: round trip {tags} != {gold}"
        assert not log["removed"], f"{split} sentence {k + 1}: gold links removed by constraints"
    print(f"ok  {split}: {len(data)} sentences (clean, constraint table, G round trip)")


class _BareCRFTagger(du.CRFMultitaskTagger):
    """CRFMultitaskTagger without the encoder, for testing decoding only."""

    def __init__(self, num_tags):
        torch.nn.Module.__init__(self)
        self.crf = du.CRF(num_tags, batch_first=True)
        self.num_mwe_tags = num_tags
        self.register_buffer("allowed_trans", torch.ones(num_tags, num_tags, dtype=torch.bool), persistent=False)
        self.register_buffer("allowed_start", torch.ones(num_tags, dtype=torch.bool), persistent=False)
        self.register_buffer("allowed_end", torch.ones(num_tags, dtype=torch.bool), persistent=False)


def check_viterbi():
    torch.manual_seed(0)
    id2mwe = dict(enumerate(sorted(["B", "I", "O", "b", "i", "o"])))
    m = _BareCRFTagger(len(id2mwe)).eval()
    em = torch.randn(256, 25, len(id2mwe))
    mask = torch.ones(256, 25, dtype=torch.bool)
    for b in range(256):
        mask[b, int(torch.randint(1, 26, (1,))):] = False
    ref = m.crf.decode(em, mask=mask)
    assert m._constrained_viterbi(em, mask) == ref, "unmasked Viterbi differs from torchcrf decode"
    m.enable_constrained_decoding(id2mwe)
    con = m._constrained_viterbi(em, mask)
    to_tags = lambda seq: [id2mwe[i] for i in seq]
    for c, r in zip(con, ref):
        assert is_valid(to_tags(c)), "constrained path is invalid"
        if is_valid(to_tags(r)):
            assert c == r, "constrained path differs from a valid unconstrained path"
    n_changed = sum(c != r for c, r in zip(con, ref))
    print(f"ok  B+c Viterbi: 256 random sequences, {n_changed} corrected, all outputs valid")


def check_group_counts():
    res = ge.score(DATA / "dimsum16.test", DATA / "dimsum16.test")
    assert res["all"]["gold"] == 837 and res["gappy"]["gold"] == 36, res
    print("ok  gappy_eval: 837 gold groups, 36 gappy")


if __name__ == "__main__":
    if not DATA.exists():
        sys.exit("Run from the repo root with dimsum-data/ present.")
    check_gold("dimsum16.train")
    check_gold("dimsum16.test")
    check_viterbi()
    check_group_counts()
    print("all checks passed")
