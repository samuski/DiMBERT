#!/usr/bin/env python3
"""
Unified DiMSUM runner for VS Code/local Python and Google Colab.

Supports:
  - BERT/RoBERTa/DeBERTa backbones through Hugging Face AutoModel/AutoTokenizer
  - linear multitask heads: MWE + supersense
  - optional CRF decoding for the MWE head
  - train/dev split from training data only
  - prediction file writing in DiMSUM format
  - optional official dimsumeval.py call

Example local/VS Code:
  python dimsum_unified.py --data_dir ./dimsum-data/data --eval_file ./dimsum-data/eval/dimsumeval.py \
    --model_name roberta-base --architecture linear --epochs 3 --batch_size 16

Example Colab:
  !python dimsum_unified.py --data_dir /content/drive/MyDrive/DiMSUM/data \
    --eval_file /content/drive/MyDrive/DiMSUM/eval/dimsumeval.py \
    --model_name microsoft/deberta-v3-small --architecture mtl_crf --epochs 3
"""

from __future__ import annotations

import argparse
from html import parser
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging

try:
    from torchcrf import CRF
except Exception:  # pragma: no cover
    CRF = None

try:
    from sklearn.metrics import f1_score
except Exception:  # pragma: no cover
    f1_score = None

# (word, mwe_tag, supersense, mwe_parent) where mwe_parent is the 1-based index
# of the nearest preceding token of the same MWE (DiMSUM column 6), or 0.
Sentence = List[Tuple[str, str, Optional[str], int]]

# DiMSUM 6-tag MWE scheme (no strength distinction): B/I = outer expression,
# b/i = expression inside a gap, o = single word inside a gap, O = outside.
# Valid tag bigrams for constrained decoding (cf. Schneider et al. 2014;
# every bigram observed in dimsum16.train and dimsum16.test is in this set).
MWE_ALLOWED_NEXT = {
    "O": {"O", "B"},
    "B": {"I", "o", "b"},
    "I": {"O", "B", "I", "o", "b"},
    "o": {"o", "b", "I"},
    "b": {"i"},
    "i": {"i", "o", "b", "I"},
}
MWE_ALLOWED_START = {"O", "B"}
MWE_ALLOWED_END = {"O", "I"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def running_in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def maybe_mount_drive() -> None:
    if not running_in_colab():
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Google Drive mount skipped: {exc}")


def parse_dimsum_file(file_path: Path) -> List[Sentence]:
    """Parse DiMSUM CoNLL-style file into sentences of (word, mwe_tag, supersense)."""
    sentences: List[Sentence] = []
    current: Sentence = []

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue

            cols = line.split("\t")
            if len(cols) < 5:
                continue

            word = cols[1]
            mwe_tag = cols[4] if cols[4] else "O"
            sup_tag = cols[7].strip() if len(cols) > 7 and cols[7].strip() else None
            parent = int(cols[5]) if len(cols) > 5 and cols[5].strip().isdigit() else 0
            current.append((word, mwe_tag, sup_tag, parent))

    if current:
        sentences.append(current)
    return sentences


def parse_dimsum_raw(file_path: Path) -> List[List[str]]:
    """Raw token lines per sentence, using the same filtering as parse_dimsum_file
    so that index k refers to the same sentence in both."""
    blocks: List[List[str]] = []
    current: List[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current:
                    blocks.append(current)
                    current = []
                continue
            if len(line.split("\t")) < 5:
                continue
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def write_dimsum_raw(blocks: Sequence[Sequence[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for block in blocks:
            for line in block:
                f.write(line + "\n")
            f.write("\n")


def build_vocabs(train_data: Sequence[Sentence]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build label vocabularies from training data only. This avoids test-label leakage."""
    mwe_vocab = {"O"}
    sup_vocab = {"O"}
    for sentence in train_data:
        for _, mwe, sup, _parent in sentence:
            mwe_vocab.add(mwe if mwe else "O")
            if sup:
                sup_vocab.add(sup)
    return ({tag: i for i, tag in enumerate(sorted(mwe_vocab))},
            {tag: i for i, tag in enumerate(sorted(sup_vocab))})


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {idx: tag for tag, idx in vocab.items()}


class DiMSUMDataset(Dataset):
    def __init__(self, data: Sequence[Sentence], tokenizer, max_len: int,
                 mwe2id: Dict[str, int], sup2id: Dict[str, int]):
        self.data = list(data)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mwe2id = mwe2id
        self.sup2id = sup2id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        sentence = self.data[index]
        words = [x[0] for x in sentence]
        mwe_tags = [x[1] for x in sentence]
        sup_tags = [x[2] for x in sentence]
        parents = [x[3] if len(x) > 3 else 0 for x in sentence]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids()

        mwe_label_ids: List[int] = []
        sup_label_ids: List[int] = []
        first_subword_mask: List[int] = []
        # Parent-selection target (used only by --architecture mtl_parent):
        # at each word's first subword, the subword position of its MWE parent's
        # first subword, or its own position meaning "no parent".
        parent_targets: List[int] = []
        word_first_pos: Dict[int, int] = {}

        prev_word_idx = None
        for pos, word_idx in enumerate(word_ids):
            if word_idx is None:
                mwe_label_ids.append(self.mwe2id["O"])
                sup_label_ids.append(-100)
                first_subword_mask.append(0)
                parent_targets.append(-100)
            elif word_idx != prev_word_idx:
                mwe_label_ids.append(self.mwe2id.get(mwe_tags[word_idx], self.mwe2id["O"]))
                sup = sup_tags[word_idx]
                sup_label_ids.append(self.sup2id.get(sup, self.sup2id["O"]) if sup else self.sup2id["O"])
                first_subword_mask.append(1)
                word_first_pos[word_idx] = pos
                parent_word = parents[word_idx] - 1  # 1-based -> 0-based; -1 means none
                if 0 <= parent_word < word_idx and parent_word in word_first_pos:
                    parent_targets.append(word_first_pos[parent_word])
                else:
                    parent_targets.append(pos)
            else:
                mwe_label_ids.append(self.mwe2id["O"])
                sup_label_ids.append(-100)
                first_subword_mask.append(0)
                parent_targets.append(-100)
            prev_word_idx = word_idx

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return (
            item["input_ids"],
            item["attention_mask"],
            torch.tensor(first_subword_mask, dtype=torch.bool),
            torch.tensor(mwe_label_ids, dtype=torch.long),
            torch.tensor(sup_label_ids, dtype=torch.long),
            torch.tensor(parent_targets, dtype=torch.long),
        )


class LinearMultitaskTagger(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_mwe_tags: int,
        num_sup_tags: int,
        dropout: float = 0.1,
        mwe_loss_weight: float = 1.0,
        sup_loss_weight: float = 1.0,
    ):
        super().__init__()
        transformers_logging.set_verbosity_error()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
        transformers_logging.set_verbosity_warning()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.mwe_head = nn.Linear(hidden, num_mwe_tags)
        self.sup_head = nn.Linear(hidden, num_sup_tags)
        self.num_mwe_tags = num_mwe_tags
        self.num_sup_tags = num_sup_tags
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.mwe_loss_weight = mwe_loss_weight
        self.sup_loss_weight = sup_loss_weight

    def forward(self, input_ids, attention_mask, first_subword_mask=None, mwe_tags=None, sup_tags=None, parent_tags=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out = self.dropout(out)
        out = out.to(self.mwe_head.weight.dtype)
        mwe_logits = self.mwe_head(out)
        sup_logits = self.sup_head(out)
        mwe_preds = torch.argmax(mwe_logits, dim=-1).tolist()
        sup_preds = torch.argmax(sup_logits, dim=-1)

        if mwe_tags is not None and sup_tags is not None:
            # Only score first subwords for MWE; ignore special/padding/trailing subwords.
            masked_mwe_tags = mwe_tags.masked_fill(~first_subword_mask, -100)
            mwe_loss = self.loss_fn(mwe_logits.view(-1, self.num_mwe_tags), masked_mwe_tags.view(-1))
            sup_loss = self.loss_fn(sup_logits.view(-1, self.num_sup_tags), sup_tags.view(-1))
            loss = (
                self.mwe_loss_weight * mwe_loss
                + self.sup_loss_weight * sup_loss
            )
            return loss, mwe_preds, sup_preds
        return mwe_preds, sup_preds


class CRFMultitaskTagger(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_mwe_tags: int,
        num_sup_tags: int,
        dropout: float = 0.1,
        mwe_loss_weight: float = 1.0,
        sup_loss_weight: float = 1.0,
    ):
        super().__init__()
        if CRF is None:
            raise RuntimeError("pytorch-crf is required for --architecture mtl_crf. Install with: pip install pytorch-crf")
        transformers_logging.set_verbosity_error()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
        transformers_logging.set_verbosity_warning()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.mwe_head = nn.Linear(hidden, num_mwe_tags)
        self.sup_head = nn.Linear(hidden, num_sup_tags)
        self.crf = CRF(num_mwe_tags, batch_first=True)
        self.num_mwe_tags = num_mwe_tags
        self.num_sup_tags = num_sup_tags
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.mwe_loss_weight = mwe_loss_weight
        self.sup_loss_weight = sup_loss_weight
        # Constrained decoding (B+c). Training is unchanged; only Viterbi
        # decoding is restricted to valid DiMSUM tag sequences.
        self.constrained = False
        self.register_buffer("allowed_trans", torch.ones(num_mwe_tags, num_mwe_tags, dtype=torch.bool), persistent=False)
        self.register_buffer("allowed_start", torch.ones(num_mwe_tags, dtype=torch.bool), persistent=False)
        self.register_buffer("allowed_end", torch.ones(num_mwe_tags, dtype=torch.bool), persistent=False)

    def enable_constrained_decoding(self, id2mwe: Dict[int, str]) -> None:
        """Mask invalid tag bigrams / start / end tags at decoding time."""
        n = self.num_mwe_tags
        trans = torch.zeros(n, n, dtype=torch.bool)
        start = torch.zeros(n, dtype=torch.bool)
        end = torch.zeros(n, dtype=torch.bool)
        for a in range(n):
            ta = id2mwe[a]
            start[a] = ta in MWE_ALLOWED_START
            end[a] = ta in MWE_ALLOWED_END
            for b in range(n):
                trans[a, b] = id2mwe[b] in MWE_ALLOWED_NEXT.get(ta, set())
        device = self.crf.transitions.device
        self.allowed_trans = trans.to(device)
        self.allowed_start = start.to(device)
        self.allowed_end = end.to(device)
        self.constrained = True

    def _constrained_viterbi(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi over the learned CRF scores with invalid transitions removed.
        emissions: [B, L, K]; mask: [B, L] (contiguous prefix of True)."""
        neg = -1e4
        trans = self.crf.transitions.masked_fill(~self.allowed_trans, neg)
        start = self.crf.start_transitions.masked_fill(~self.allowed_start, neg)
        end = self.crf.end_transitions.masked_fill(~self.allowed_end, neg)
        batch_size, seq_len, _ = emissions.shape
        results: List[List[int]] = []
        for b in range(batch_size):
            length = int(mask[b].sum())
            if length == 0:
                results.append([])
                continue
            em = emissions[b, :length]
            score = start + em[0]
            backpointers = []
            for t in range(1, length):
                total = score.unsqueeze(1) + trans + em[t].unsqueeze(0)  # [prev, cur]
                score, idx = total.max(dim=0)
                backpointers.append(idx)
            score = score + end
            best = int(score.argmax())
            path = [best]
            for idx in reversed(backpointers):
                best = int(idx[best])
                path.append(best)
            path.reverse()
            results.append(path)
        return results

    @staticmethod
    def _pack_crf_inputs(
        mwe_logits: torch.Tensor,
        first_subword_mask: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        mwe_tags: Optional[torch.Tensor] = None,
    ):
        if first_subword_mask is None:
            return mwe_logits, mwe_tags, attention_mask.bool()

        batch_size, _, num_tags = mwe_logits.shape
        packed_positions = []
        max_len = 0

        for i in range(batch_size):
            positions = torch.where(first_subword_mask[i])[0]
            packed_positions.append(positions)
            max_len = max(max_len, int(positions.numel()))

        packed_logits = mwe_logits.new_zeros((batch_size, max_len, num_tags))
        packed_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=mwe_logits.device)
        packed_tags = None
        if mwe_tags is not None:
            packed_tags = mwe_tags.new_zeros((batch_size, max_len))

        for i, positions in enumerate(packed_positions):
            length = int(positions.numel())
            packed_logits[i, :length] = mwe_logits[i, positions]
            packed_mask[i, :length] = True
            if packed_tags is not None:
                packed_tags[i, :length] = mwe_tags[i, positions]

        return packed_logits, packed_tags, packed_mask

    def forward(self, input_ids, attention_mask, first_subword_mask=None, mwe_tags=None, sup_tags=None, parent_tags=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out = self.dropout(out)
        out = out.to(self.mwe_head.weight.dtype)
        mwe_logits = self.mwe_head(out)
        sup_logits = self.sup_head(out)

        # torchcrf expects each sequence mask to be contiguous. Pack first-subword
        # emissions into word-level CRF sequences before scoring/decoding.
        crf_logits, crf_tags, crf_mask = self._pack_crf_inputs(
            mwe_logits,
            first_subword_mask,
            attention_mask,
            mwe_tags,
        )

        if self.constrained:
            mwe_preds = None if self.training else self._constrained_viterbi(crf_logits, crf_mask)
        else:
            mwe_preds = self.crf.decode(crf_logits, mask=crf_mask)
        sup_preds = torch.argmax(sup_logits, dim=-1)

        if mwe_tags is not None and sup_tags is not None:
            mwe_loss = -self.crf(crf_logits, crf_tags, mask=crf_mask, reduction="mean")
            sup_loss = self.loss_fn(sup_logits.view(-1, self.num_sup_tags), sup_tags.view(-1))

            loss = (
                self.mwe_loss_weight * mwe_loss
                + self.sup_loss_weight * sup_loss
            )
            return loss, mwe_preds, sup_preds
        return mwe_preds, sup_preds


class ParentMultitaskTagger(nn.Module):
    """
    MWE identification as parent selection (condition G).

    DiMSUM column 6 links every non-initial MWE token to the nearest preceding
    token of the same MWE, so each MWE is a left-to-right chain. For every word
    j the model scores each preceding word i as j's parent, plus "none"
    (encoded as j pointing to itself), with a biaffine scorer over first-subword
    encodings, and is trained with cross-entropy over these candidates.
    Candidate parent-child links are scored directly as token pairs, instead of
    being implied by a sequence of local tags.

    The supersense head and loss are the same as in the CRF tagger.
    """

    def __init__(
        self,
        model_name: str,
        num_sup_tags: int,
        dropout: float = 0.1,
        mwe_loss_weight: float = 1.0,
        sup_loss_weight: float = 1.0,
        arc_dim: int = 256,
    ):
        super().__init__()
        transformers_logging.set_verbosity_error()
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
        transformers_logging.set_verbosity_warning()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sup_head = nn.Linear(hidden, num_sup_tags)
        self.child_mlp = nn.Sequential(nn.Linear(hidden, arc_dim), nn.GELU(), nn.Dropout(dropout))
        self.parent_mlp = nn.Sequential(nn.Linear(hidden, arc_dim), nn.GELU(), nn.Dropout(dropout))
        self.arc_weight = nn.Parameter(torch.zeros(arc_dim, arc_dim))
        self.arc_parent_bias = nn.Linear(arc_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.arc_weight)
        self.num_sup_tags = num_sup_tags
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.mwe_loss_weight = mwe_loss_weight
        self.sup_loss_weight = sup_loss_weight

    def forward(self, input_ids, attention_mask, first_subword_mask=None, mwe_tags=None, sup_tags=None, parent_tags=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out = self.dropout(out)
        out = out.to(self.sup_head.weight.dtype)
        sup_logits = self.sup_head(out)
        sup_preds = torch.argmax(sup_logits, dim=-1)

        child = self.child_mlp(out)    # [B, T, d]
        parent = self.parent_mlp(out)  # [B, T, d]
        # scores[b, t, s] = score of position s being the parent of position t
        scores = torch.einsum("btd,de,bse->bts", child, self.arc_weight, parent)
        scores = scores + self.arc_parent_bias(parent).squeeze(-1).unsqueeze(1)

        seq_len = input_ids.size(1)
        if first_subword_mask is None:
            first_subword_mask = attention_mask.bool()
        positions = torch.arange(seq_len, device=input_ids.device)
        earlier = positions.unsqueeze(0) < positions.unsqueeze(1)          # [T, T]: s < t
        self_pos = positions.unsqueeze(0) == positions.unsqueeze(1)       # [T, T]: s == t ("none")
        candidates = (earlier.unsqueeze(0) & first_subword_mask.unsqueeze(1)) | self_pos.unsqueeze(0)
        scores = scores.masked_fill(~candidates, -1e4)
        parent_logprobs = torch.log_softmax(scores, dim=-1)

        if sup_tags is not None and parent_tags is not None:
            arc_loss = self.loss_fn(scores.view(-1, seq_len), parent_tags.view(-1))
            sup_loss = self.loss_fn(sup_logits.view(-1, self.num_sup_tags), sup_tags.view(-1))
            loss = self.mwe_loss_weight * arc_loss + self.sup_loss_weight * sup_loss
            return loss, parent_logprobs, sup_preds
        return parent_logprobs, sup_preds


def decode_parent_links(logprobs: torch.Tensor, valid_indices: torch.Tensor):
    """
    Turn parent-selection scores for one sentence into DiMSUM MWE tags.

    logprobs: [T, T] log-probabilities (row = child position, column = parent
              position; the diagonal is "none").
    valid_indices: first-subword positions of the words, in order.

    Steps:
      1. Each word takes its highest-scoring parent (unconstrained top-1).
      2. At most one child per parent: keep the highest-scoring child, set the
         others to "none".
      3. Structural constraint (DiMSUM): two MWEs may overlap in span only if
         one lies entirely inside a single gap of the other and the inner one
         is contiguous. On a violation, drop the lowest-scoring link among the
         two MWEs and repeat.
      4. Convert the remaining chains to O/o/B/b/I/i tags.

    Returns (tags, log) where log records unconstrained links, final links,
    removed links with reason and score margin, and the none rate.
    """
    pos = [int(p) for p in valid_indices]
    n = len(pos)
    lp = logprobs.detach().float().cpu()

    parent: List[int] = [-1] * n
    score: List[float] = [0.0] * n
    none_score: List[float] = [0.0] * n
    for j in range(n):
        none_score[j] = float(lp[pos[j], pos[j]])
        best_i, best_s = -1, none_score[j]
        for i in range(j):
            s = float(lp[pos[j], pos[i]])
            if s > best_s:
                best_i, best_s = i, s
        parent[j], score[j] = best_i, best_s
    unconstrained = list(parent)
    removed: List[Dict[str, object]] = []

    # 2. at most one child per parent
    children: Dict[int, List[int]] = {}
    for j, p in enumerate(parent):
        if p >= 0:
            children.setdefault(p, []).append(j)
    for p, kids in children.items():
        if len(kids) <= 1:
            continue
        keep = max(kids, key=lambda k: score[k])
        for k in kids:
            if k != keep:
                removed.append({
                    "child": k, "parent": p, "score": score[k], "none_score": none_score[k],
                    "reason": "one_child", "retained_child": keep, "retained_score": score[keep],
                    "margin": score[keep] - score[k], "cross_gap": k - p > 1,
                })
                parent[k] = -1

    def build_groups() -> List[List[int]]:
        child_of = {p: j for j, p in enumerate(parent) if p >= 0}
        groups = []
        for j in range(n):
            if parent[j] == -1 and j in child_of:
                chain = [j]
                while chain[-1] in child_of:
                    chain.append(child_of[chain[-1]])
                groups.append(chain)
        return groups

    def inside_one_gap(inner: List[int], outer: List[int]) -> bool:
        for a, b in zip(outer, outer[1:]):
            if all(a < x < b for x in inner):
                return True
        return False

    def contiguous(g: List[int]) -> bool:
        return g[-1] - g[0] + 1 == len(g)

    # 3. structural constraint
    while True:
        groups = build_groups()
        violation = None
        for gi in range(len(groups)):
            for hi in range(gi + 1, len(groups)):
                g, h = groups[gi], groups[hi]
                if g[-1] < h[0] or h[-1] < g[0]:
                    continue  # disjoint spans
                ok = (inside_one_gap(h, g) and contiguous(h)) or (inside_one_gap(g, h) and contiguous(g))
                if not ok:
                    violation = (g, h)
                    break
            if violation:
                break
        if not violation:
            break
        g, h = violation
        links = [k for k in g[1:] + h[1:]]  # children (each non-first member has one parent link)
        worst = min(links, key=lambda k: score[k])
        others = [k for k in links if k != worst]
        removed.append({
            "child": worst, "parent": parent[worst], "score": score[worst], "none_score": none_score[worst],
            "reason": "structure",
            "retained_min_score": min(score[k] for k in others) if others else None,
            "margin": (min(score[k] for k in others) - score[worst]) if others else None,
            "cross_gap": worst - parent[worst] > 1,
        })
        parent[worst] = -1

    # 4. chains -> tags
    groups = build_groups()
    tags = ["O"] * n
    inner_groups = [g for g in groups if any(inside_one_gap(g, o) for o in groups if o is not g)]
    outer_groups = [g for g in groups if g not in inner_groups]
    for g in outer_groups:
        tags[g[0]] = "B"
        for k in g[1:]:
            tags[k] = "I"
        members = set(g)
        for k in range(g[0] + 1, g[-1]):
            if k not in members:
                tags[k] = "o"
    for g in inner_groups:
        tags[g[0]] = "b"
        for k in g[1:]:
            tags[k] = "i"

    log = {
        "n_words": n,
        "unconstrained_parents": unconstrained,
        "final_parents": list(parent),
        "unconstrained_none": sum(1 for p in unconstrained if p < 0),
        "removed": removed,
    }
    return tags, log


def make_model(
    architecture: str,
    model_name: str,
    num_mwe_tags: int,
    num_sup_tags: int,
    dropout: float,
    mwe_loss_weight: float,
    sup_loss_weight: float,
):
    if architecture == "linear":
        return LinearMultitaskTagger(
            model_name,
            num_mwe_tags,
            num_sup_tags,
            dropout,
            mwe_loss_weight,
            sup_loss_weight,
        )
    if architecture == "mtl_crf":
        return CRFMultitaskTagger(
            model_name,
            num_mwe_tags,
            num_sup_tags,
            dropout,
            mwe_loss_weight,
            sup_loss_weight,
        )
    if architecture == "mtl_parent":
        return ParentMultitaskTagger(
            model_name,
            num_sup_tags,
            dropout,
            mwe_loss_weight,
            sup_loss_weight,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def split_train_dev(data: Sequence[Sentence], dev_split: float, seed: int):
    items = list(data)
    random.Random(seed).shuffle(items)
    split = int(len(items) * (1.0 - dev_split))
    return items[:split], items[split:]


def train_one(
    model,
    loader,
    device,
    epochs: int,
    lr: float,
    grad_clip: float,
    on_epoch_end=None,
):
    """Train for a fixed number of epochs. If on_epoch_end is given, it is
    called after every epoch with (epoch, model) and may return a dict of
    metrics that is merged into that epoch's loss_history row."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        total = 0.0

        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")

        for batch in bar:
            batch = [x.to(device) for x in batch]

            optimizer.zero_grad(set_to_none=True)

            loss, _, _ = model(*batch[:3], mwe_tags=batch[3], sup_tags=batch[4], parent_tags=batch[5])

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total += float(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total / max(len(loader), 1)

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if on_epoch_end is not None:
            row.update(on_epoch_end(epoch + 1, model) or {})

        loss_history.append(row)

        print(
            f"epoch {epoch + 1}: "
            f"avg_train_loss={avg_train_loss:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    return model, loss_history


def macro_f1(y_true, y_pred) -> float:
    if not y_true:
        return 0.0
    if f1_score is None:
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        return correct / len(y_true)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def decode_mwe_predictions(architecture: str, raw_preds, valid_indices: torch.Tensor, id2mwe: Dict[int, str],
                           decode_logs: Optional[list] = None) -> List[str]:
    if architecture == "mtl_parent":
        tags, log = decode_parent_links(raw_preds, valid_indices)
        if decode_logs is not None:
            decode_logs.append(log)
        return tags
    if architecture == "mtl_crf":
        return [id2mwe[raw_preds[j]] for j in range(len(valid_indices))]
    return [id2mwe[raw_preds[int(idx)]] for idx in valid_indices]


def evaluate_dev(model, loader, device, architecture: str, id2mwe: Dict[int, str], id2sup: Dict[int, str]):
    model.eval()
    total_loss = 0.0
    flat_mwe_t, flat_mwe_p = [], []
    flat_sup_t, flat_sup_p = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]
            input_ids, attention_mask, first_mask, mwe_tags, sup_tags, parent_tags = batch
            loss, mwe_preds, sup_preds = model(input_ids, attention_mask, first_mask, mwe_tags, sup_tags, parent_tags)
            total_loss += float(loss.item())
            for i in range(input_ids.size(0)):
                valid_indices = torch.where(first_mask[i])[0]
                mwe_p = decode_mwe_predictions(architecture, mwe_preds[i], valid_indices, id2mwe)
                mwe_t = [id2mwe[int(mwe_tags[i, idx])] for idx in valid_indices]
                sup_p_raw = [id2sup[int(sup_preds[i, idx])] for idx in valid_indices]
                sup_t_raw = [id2sup[int(sup_tags[i, idx])] if int(sup_tags[i, idx]) != -100 else "O" for idx in valid_indices]
                flat_mwe_t.extend(mwe_t)
                flat_mwe_p.extend(mwe_p)
                flat_sup_t.extend(sup_t_raw)
                flat_sup_p.extend(sup_p_raw)
    return {
        "dev_loss": total_loss / max(len(loader), 1),
        "mwe_macro_f1": macro_f1(flat_mwe_t, flat_mwe_p),
        "sup_macro_f1": macro_f1(flat_sup_t, flat_sup_p),
    }


def normalize_mwe_tag(tag: Optional[str]) -> str:
    """
    Collapse DiMSUM's gappy lowercase MWE tags into simple BIO.

    The official DiMSUM evaluator supports lowercase b/i/o only for valid
    discontinuous MWEs. This baseline does not explicitly model gappy MWEs,
    so writing lowercase sequences can create invalid outputs such as "bio".
    For stable evaluation, predicted MWEs are normalized to standard B/I/O.
    """
    if tag in {"B", "b"}:
        return "B"
    if tag in {"I", "i"}:
        return "I"
    return "O"


def clean_mwe_tags(tags: Sequence[str]) -> List[str]:
    """
    DiMSUM-aware cleanup for the official 6-tag scheme.

    Keeps: B, I, O, b, i, o

    Valid examples:
      B I
      B I I
      B o o I
      B b i I

    Invalid examples repaired:
      B O        -> O O
      B o O      -> O O O
      B o o i I  -> B o o o I
      I O        -> O O
      o I        -> O O
    """
    valid = {"B", "I", "O", "b", "i", "o"}
    tags = [t if t in valid else "O" for t in tags]
    n = len(tags)
    out = ["O"] * n

    i = 0
    while i < n:
        t = tags[i]

        if t == "O":
            i += 1
            continue

        # These cannot start an outer DiMSUM MWE chunk.
        if t in {"I", "i", "o"}:
            i += 1
            continue

        # Uppercase B starts the outer MWE chunk.
        # Lowercase b at the start is repaired as uppercase B.
        if t in {"B", "b"}:
            start = i
            out[start] = "B"

            j = i + 1
            found_upper_I = False
            lower_b_active = False

            while j < n:
                tj = tags[j]

                if tj == "O":
                    break

                if tj == "B":
                    # New chunk begins. Stop current chunk here.
                    break

                if tj == "I":
                    out[j] = "I"
                    found_upper_I = True
                    lower_b_active = False
                    j += 1
                    continue

                if tj == "o":
                    out[j] = "o"
                    lower_b_active = False
                    j += 1
                    continue

                if tj == "b":
                    # Lowercase b starts a nested/weak expression inside a larger MWE.
                    # It only stays b if followed by at least one i.
                    if j + 1 < n and tags[j + 1] == "i":
                        out[j] = "b"
                        lower_b_active = True
                    else:
                        out[j] = "o"
                        lower_b_active = False
                    j += 1
                    continue

                if tj == "i":
                    if lower_b_active:
                        out[j] = "i"
                    else:
                        # Bare lowercase i is invalid after o/I/B.
                        # Treat it as a gap marker inside the outer MWE.
                        out[j] = "o"
                    j += 1
                    continue

                break

            if not found_upper_I:
                # Singleton B or B...o without a final I is not a valid MWE.
                for k in range(start, j):
                    out[k] = "O"
            else:
                # Valid DiMSUM chunks must end with an uppercase I.
                last_upper_I = max(k for k in range(start + 1, j) if out[k] == "I")
                for k in range(last_upper_I + 1, j):
                    out[k] = "O"

            i = j
            continue

        i += 1

    return out


def write_prediction_file(
    test_file: Path,
    pred_file: Path,
    mwe_preds: List[List[str]],
    sup_preds: List[List[Optional[str]]],
):
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_mwe = [clean_mwe_tags(x) for x in mwe_preds]

    with test_file.open("r", encoding="utf-8") as f_in, pred_file.open("w", encoding="utf-8") as f_out:
        sent_idx, word_idx = 0, 0
        strong_head = "0"  # head for B ... I
        weak_head = "0"    # head for b ... i

        for raw_line in f_in:
            line = raw_line.rstrip("\n")

            if not line.strip():
                f_out.write("\n")
                sent_idx += 1
                word_idx = 0
                strong_head = "0"
                weak_head = "0"
                continue

            cols = line.split("\t")
            while len(cols) < 8:
                cols.append("")

            if sent_idx < len(cleaned_mwe) and word_idx < len(cleaned_mwe[sent_idx]):
                mwe = cleaned_mwe[sent_idx][word_idx]
                sup = sup_preds[sent_idx][word_idx]

                if mwe == "B":
                    strong_head = cols[0]
                    weak_head = "0"
                    cols[4] = "B"
                    cols[5] = "0"
                    cols[6] = ""
                    cols[7] = sup if sup and sup != "O" else ""

                elif mwe == "I":
                    cols[4] = "I"
                    cols[5] = strong_head if strong_head != "0" else "0"
                    cols[6] = ""
                    cols[7] = ""

                elif mwe == "o":
                    # Gap token inside a discontinuous MWE.
                    # It is not linked to the MWE group, but it may have its own supersense.
                    cols[4] = "o"
                    cols[5] = "0"
                    cols[6] = ""
                    cols[7] = sup if sup and sup != "O" else ""

                elif mwe == "b":
                    weak_head = cols[0]
                    cols[4] = "b"
                    cols[5] = "0"
                    cols[6] = ""
                    cols[7] = sup if sup and sup != "O" else ""

                elif mwe == "i":
                    cols[4] = "i"
                    cols[5] = weak_head if weak_head != "0" else "0"
                    cols[6] = ""
                    cols[7] = ""

                else:
                    strong_head = "0"
                    weak_head = "0"
                    cols[4] = "O"
                    cols[5] = "0"
                    cols[6] = ""
                    cols[7] = sup if sup and sup != "O" else ""
            else:
                # Token beyond the model's predictions (sentence truncated at
                # max_len): predict nothing, rather than copying the gold
                # MWE/supersense columns through from the reference file.
                cols[4] = "O"
                cols[5] = "0"
                cols[6] = ""
                cols[7] = ""

            f_out.write("\t".join(cols) + "\n")
            word_idx += 1


def predict_and_write(model, loader, device, architecture: str, id2mwe, id2sup, test_file: Path, pred_file: Path,
                      decode_logs: Optional[list] = None) -> Dict[str, int]:
    """Predict, write a DiMSUM-format prediction file, and return diagnostics:
    number of sentences whose raw MWE tag output was structurally invalid
    (changed by clean_mwe_tags before writing)."""
    model.eval()
    all_mwe_preds: List[List[str]] = []
    all_sup_preds: List[List[Optional[str]]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            batch = [x.to(device) for x in batch]
            input_ids, attention_mask, first_mask = batch[:3]
            mwe_preds, sup_preds = model(input_ids, attention_mask, first_mask)
            for i in range(input_ids.size(0)):
                valid_indices = torch.where(first_mask[i])[0]
                all_mwe_preds.append(decode_mwe_predictions(architecture, mwe_preds[i], valid_indices, id2mwe, decode_logs))
                raw_sup = [id2sup[int(sup_preds[i, idx])] for idx in valid_indices]
                all_sup_preds.append([x if x != "O" else None for x in raw_sup])
    invalid = sum(1 for tags in all_mwe_preds if clean_mwe_tags(tags) != list(tags))
    write_prediction_file(test_file, pred_file, all_mwe_preds, all_sup_preds)
    return {"sentences": len(all_mwe_preds), "invalid_mwe_sentences_before_repair": invalid}


def run_official_eval(eval_file: Optional[Path], gold_file: Path, pred_file: Path) -> str:
    if not eval_file or not eval_file.exists():
        return "Official evaluator not found; skipped."

    cmd = [sys.executable, str(eval_file), "-C", str(gold_file), str(pred_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if result.returncode != 0:
        raise RuntimeError(
            f"Official evaluator failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}\n\n{output}"
        )

    return output


def _ratio_decimal_to_percent(value: str) -> Optional[float]:
    try:
        return float(value) * 100.0
    except Exception:
        return None


def parse_official_scores(text: str) -> Dict[str, float]:
    """
    Parse official dimsumeval.py summary lines.

    Expected lines look like:
      MWEs: P=145/537=0.2700 R=145/1115=0.1300 F=17.55%
      Supersenses: P=1498/3851=0.3890 R=1498/4745=0.3157 F=34.85%
      Combined: Acc=11493/16500=0.6965 P=1643/4388=0.3744 R=1643/5860=0.2804 F=32.06%

    Values are returned as percentages for easy table generation.
    """
    scores: Dict[str, float] = {}

    task_map = {
        "MWEs": "mwe",
        "Supersenses": "sup",
        "Combined": "combined",
    }

    for raw in text.splitlines():
        line = raw.strip()
        for label, prefix in task_map.items():
            if not line.startswith(label + ":"):
                continue

            acc = re.search(r"Acc=[^=\s]+=[ ]*([0-9.]+)", line)
            p = re.search(r"\bP=[^=\s]+=[ ]*([0-9.]+)", line)
            r = re.search(r"\bR=[^=\s]+=[ ]*([0-9.]+)", line)
            f = re.search(r"\bF=([0-9.]+)%", line)

            if acc:
                val = _ratio_decimal_to_percent(acc.group(1))
                if val is not None:
                    scores[f"official_{prefix}_acc"] = val
            if p:
                val = _ratio_decimal_to_percent(p.group(1))
                if val is not None:
                    scores[f"official_{prefix}_precision"] = val
            if r:
                val = _ratio_decimal_to_percent(r.group(1))
                if val is not None:
                    scores[f"official_{prefix}_recall"] = val
            if f:
                scores[f"official_{prefix}_f1"] = float(f.group(1))

    return scores


@dataclass
class RunResult:
    architecture: str
    model_name: str
    lr: float
    epochs: int
    batch_size: int
    dev_loss: float
    mwe_macro_f1: float
    sup_macro_f1: float
    pred_file: str
    model_file: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("./dimsum-data/data"))
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument("--eval_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("./runs"))
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--architecture", choices=["linear", "mtl_crf", "mtl_parent"], default="linear",
                        help="mtl_crf = CRF MWE tagger (B); mtl_parent = parent-selection MWE head (G).")
    parser.add_argument("--constrained_decoding", action="store_true",
                        help="mtl_crf only: restrict Viterbi decoding to valid DiMSUM tag sequences (B+c).")
    parser.add_argument("--no_dev_selection", action="store_true",
                        help="Use the final epoch instead of the checkpoint with the best dev combined F.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dev_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--mount_drive", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick_cpu", action="store_true", help="Small, CPU-friendly run for smoke testing.")
    parser.add_argument("--mwe_loss_weight", type=float, default=1.0)
    parser.add_argument("--sup_loss_weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.constrained_decoding and args.architecture != "mtl_crf":
        parser.error("--constrained_decoding requires --architecture mtl_crf")
    dev_selection = not args.no_dev_selection

    if args.mount_drive:
        maybe_mount_drive()

    if args.quick_cpu:
        args.cpu = True
        args.model_name = "distilbert-base-uncased"
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 4)
        args.max_len = min(args.max_len, 96)

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")

    train_file = args.train_file or args.data_dir / "dimsum16.train"
    test_file = args.test_file or args.data_dir / "dimsum16.test"
    if args.eval_file is None:
        candidates = [
            args.data_dir / "scripts" / "dimsumeval.py",
            args.data_dir.parent / "scripts" / "dimsumeval.py",
            args.data_dir.parent / "eval" / "dimsumeval.py",
        ]
        args.eval_file = next((c for c in candidates if c.exists()), None)

    print(f"train_file={train_file}")
    print(f"test_file={test_file}")
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("Could not find train/test files. Set --data_dir or pass --train_file and --test_file.")
    if dev_selection and (not args.eval_file or not Path(args.eval_file).exists()):
        raise FileNotFoundError("Dev checkpoint selection needs the official evaluator; pass --eval_file "
                                "(or --no_dev_selection to use the final epoch).")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_data = parse_dimsum_file(train_file)
    test_data = parse_dimsum_file(test_file)
    mwe2id, sup2id = build_vocabs(train_data)
    id2mwe, id2sup = invert_vocab(mwe2id), invert_vocab(sup2id)
    print(f"train_sentences={len(train_data)} test_sentences={len(test_data)}")
    print(f"mwe_labels={len(mwe2id)} sup_labels={len(sup2id)}")

    train_split, dev_split = split_train_dev(train_data, args.dev_split, args.seed)
    # Same permutation applied to the raw lines, to write a DiMSUM-format dev gold file.
    # (random.shuffle's permutation depends only on the seed and the list length.)
    raw_train = parse_dimsum_raw(train_file)
    assert len(raw_train) == len(train_data)
    perm = list(range(len(raw_train)))
    random.Random(args.seed).shuffle(perm)
    split_at = int(len(perm) * (1.0 - args.dev_split))
    dev_raw = [raw_train[k] for k in perm[split_at:]]
    assert [len(b) for b in dev_raw] == [len(s_) for s_ in dev_split]
    train_loader = DataLoader(DiMSUMDataset(train_split, tokenizer, args.max_len, mwe2id, sup2id), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(DiMSUMDataset(dev_split, tokenizer, args.max_len, mwe2id, sup2id), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(DiMSUMDataset(test_data, tokenizer, args.max_len, mwe2id, sup2id), batch_size=args.batch_size, shuffle=False)

    safe_model_name = args.model_name.replace("/", "__")
    arch_tag = args.architecture + ("_cd" if args.constrained_decoding else "")
    run_name = f"{arch_tag}_{safe_model_name}_lr{args.lr}_ep{args.epochs}_bs{args.batch_size}_seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_file = run_dir / "predictions.pred"
    model_file = run_dir / "model.pt"
    label_file = run_dir / "labels.json"
    dev_gold_file = run_dir / "dev.gold"
    dev_pred_file = run_dir / "dev_epoch.pred"
    write_dimsum_raw(dev_raw, dev_gold_file)

    with label_file.open("w", encoding="utf-8") as f:
        json.dump({"mwe2id": mwe2id, "sup2id": sup2id}, f, indent=2)

    start = time.time()
    model = make_model(
        args.architecture,
        args.model_name,
        len(mwe2id),
        len(sup2id),
        args.dropout,
        args.mwe_loss_weight,
        args.sup_loss_weight,
    )
    if args.constrained_decoding:
        model.enable_constrained_decoding(id2mwe)

    # Checkpoint selection on dev (official scorer): keep the epoch with the
    # highest dev combined F; ties -> higher dev supersense F; then earlier epoch.
    best = {"epoch": None, "combined": None, "sup": None, "mwe": None}

    def on_epoch_end(epoch: int, model_) -> Dict[str, float]:
        if not dev_selection:
            return {}
        predict_and_write(model_, dev_loader, device, args.architecture, id2mwe, id2sup, dev_gold_file, dev_pred_file)
        dev_scores = parse_official_scores(run_official_eval(args.eval_file, dev_gold_file, dev_pred_file))
        # A missing score (e.g. no supersenses predicted -> F=nan) ranks lowest.
        comb = dev_scores.get("official_combined_f1", float("-inf"))
        sup = dev_scores.get("official_sup_f1", float("-inf"))
        best_sup = best["sup"] if best["sup"] is not None else float("-inf")
        improved = (best["combined"] is None or comb > best["combined"]
                    or (comb == best["combined"] and sup > best_sup))
        if improved:
            best.update(epoch=epoch, combined=comb, sup=dev_scores.get("official_sup_f1"),
                        mwe=dev_scores.get("official_mwe_f1"))
            torch.save(model_.state_dict(), model_file)
        print(f"epoch {epoch}: dev combined F={comb:.2f} sup F={dev_scores.get('official_sup_f1', float('nan')):.2f} "
              f"MWE F={dev_scores.get('official_mwe_f1', float('nan')):.2f}"
              f"{'  <- best so far' if improved else ''}")
        return {
            "dev_mwe_f1": dev_scores.get("official_mwe_f1"),
            "dev_sup_f1": dev_scores.get("official_sup_f1"),
            "dev_combined_f1": dev_scores.get("official_combined_f1"),
            "selected_so_far": best["epoch"],
        }

    model, loss_history = train_one(
        model,
        train_loader,
        device,
        args.epochs,
        args.lr,
        args.grad_clip,
        on_epoch_end=on_epoch_end,
    )
    if dev_selection:
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"selected epoch {best['epoch']} (dev combined F={best['combined']:.2f})")
    else:
        torch.save(model.state_dict(), model_file)

    dev_metrics = evaluate_dev(model, dev_loader, device, args.architecture, id2mwe, id2sup)
    print("dev_metrics=", dev_metrics)

    decode_logs: Optional[list] = [] if args.architecture == "mtl_parent" else None
    pred_diag = predict_and_write(model, test_loader, device, args.architecture, id2mwe, id2sup, test_file, pred_file,
                                  decode_logs=decode_logs)
    eval_text = run_official_eval(args.eval_file, test_file, pred_file)
    with (run_dir / "official_eval.txt").open("w", encoding="utf-8") as f:
        f.write(eval_text)
    print(eval_text)

    # Exact-group gappy / contiguous scores (custom scorer).
    try:
        from gappy_eval import score as group_score, format_report as group_report
        group_scores = group_score(test_file, pred_file)
        (run_dir / "group_eval.json").write_text(json.dumps(group_scores, indent=2), encoding="utf-8")
        print(group_report(group_scores))
    except Exception as exc:  # pragma: no cover
        group_scores = {"error": str(exc)}
        print(f"group scorer failed: {exc}")

    parent_diag = None
    if decode_logs is not None:
        with (run_dir / "parent_decode_log.jsonl").open("w", encoding="utf-8") as f:
            for log in decode_logs:
                f.write(json.dumps(log) + "\n")
        removed = [r for log in decode_logs for r in log["removed"]]
        words = sum(log["n_words"] for log in decode_logs)
        unconstrained_links = sum(log["n_words"] - log["unconstrained_none"] for log in decode_logs)
        final_links = sum(sum(1 for p in log["final_parents"] if p >= 0) for log in decode_logs)
        parent_diag = {
            "words": words,
            "unconstrained_links": unconstrained_links,
            "unconstrained_cross_gap_links": sum(
                sum(1 for j, p in enumerate(log["unconstrained_parents"]) if p >= 0 and j - p > 1) for log in decode_logs),
            "final_links": final_links,
            "final_cross_gap_links": sum(
                sum(1 for j, p in enumerate(log["final_parents"]) if p >= 0 and j - p > 1) for log in decode_logs),
            "unconstrained_none_rate": round(1 - unconstrained_links / max(words, 1), 4),
            "removed_one_child": sum(1 for r in removed if r["reason"] == "one_child"),
            "removed_structure": sum(1 for r in removed if r["reason"] == "structure"),
            "removed_cross_gap": sum(1 for r in removed if r["cross_gap"]),
        }
        print("parent decoding:", json.dumps(parent_diag))

    result = RunResult(
        architecture=args.architecture,
        model_name=args.model_name,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dev_loss=dev_metrics["dev_loss"],
        mwe_macro_f1=dev_metrics["mwe_macro_f1"],
        sup_macro_f1=dev_metrics["sup_macro_f1"],
        pred_file=str(pred_file),
        model_file=str(model_file),
    )
    summary = {
        **asdict(result),
        **parse_official_scores(eval_text),
        "seconds": round(time.time() - start, 2),
        "loss_history": loss_history,
        "dropout": args.dropout,
        "grad_clip": args.grad_clip,
        "mwe_loss_weight": args.mwe_loss_weight,
        "sup_loss_weight": args.sup_loss_weight,
        "seed": args.seed,
        "constrained_decoding": args.constrained_decoding,
        "dev_selection": dev_selection,
        "selected_epoch": best["epoch"] if dev_selection else args.epochs,
        "selected_dev_combined_f1": best["combined"],
        "selected_dev_sup_f1": best["sup"],
        "selected_dev_mwe_f1": best["mwe"],
        "test_prediction_diagnostics": pred_diag,
        "group_scores": group_scores,
        "parent_decoding": parent_diag,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    loss_csv = run_dir / "loss_history.csv"
    
    with loss_csv.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,lr,dev_mwe_f1,dev_sup_f1,dev_combined_f1\n")
        for row in loss_history:
            f.write(f"{row['epoch']},{row['train_loss']},{row['lr']},"
                    f"{row.get('dev_mwe_f1', '')},{row.get('dev_sup_f1', '')},{row.get('dev_combined_f1', '')}\n")


if __name__ == "__main__":
    main()
