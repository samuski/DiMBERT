# DiMSUM Project Setup

This project trains and evaluates sequence-labeling models for the DiMSUM task. The current workflow supports:

- local Python virtual environment (`venv`)
- Docker GPU sandbox
- Hugging Face models such as BERT, RoBERTa, and DeBERTa
- official DiMSUM evaluation
- automatic report generation for analysis

## Quick start

These instructions set up the corrected CRF implementation and the Python 3-compatible
DiMSUM evaluator. Run all commands from a terminal opened at the project root.

### Get the code and data

```bash
git clone https://github.com/samuski/DiMSUM.git
cd DiMSUM
git clone --branch python3-compat https://github.com/samuski/dimsum-data.git
```

The `python3-compat` data branch contains the official DiMSUM data and a targeted
Python 3 port of its evaluation/conversion scripts. No separate evaluator patch is
needed when using that branch. Once the upstream Python 3 pull request is merged,
the official `dimsum16/dimsum-data` repository can be used directly instead.

### Option 1: local virtual environment

Python 3.11 is recommended because it matches the Docker image. On Windows
PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The Windows example installs the CUDA 12.1 PyTorch build used by this project's
Dockerfile. For a CPU-only Windows machine, replace `cu121` with `cpu`. For a
different CUDA setup, install the matching PyTorch build before installing
`requirements.txt`.

Confirm that the project and evaluator work with a short CPU smoke test:

```bash
python dimsum_unified.py --data_dir ./dimsum-data --eval_file ./dimsum-data/scripts/dimsumeval.py --quick_cpu
```

### Option 2: Docker with an NVIDIA GPU

Docker Desktop must have GPU access through WSL2, and the host must have compatible
NVIDIA drivers.

```bash
docker compose build
docker compose run --rm dimsum python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

The second command should report `CUDA: True`. The project directory is mounted at
`/workspace`, so models and reports written inside the container also appear on the
host.

### Train the corrected DeBERTa CRF model

With the virtual environment activated:

```bash
python dimsum_unified.py --data_dir ./dimsum-data --eval_file ./dimsum-data/scripts/dimsumeval.py --model_name microsoft/deberta-v3-small --architecture mtl_crf --epochs 15 --batch_size 16 --lr 2e-5 --mwe_loss_weight 3 --sup_loss_weight 2
```

Or run the same command through Docker:

```bash
docker compose run --rm dimsum python dimsum_unified.py --data_dir ./dimsum-data --eval_file ./dimsum-data/scripts/dimsumeval.py --model_name microsoft/deberta-v3-small --architecture mtl_crf --epochs 15 --batch_size 16 --lr 2e-5 --mwe_loss_weight 3 --sup_loss_weight 2
```

The run writes its checkpoint, predictions, training summary, and official evaluation
under (the folder name now includes the seed, so runs with different seeds do not
overwrite each other):

```text
runs/mtl_crf_microsoft__deberta-v3-small_lr2e-05_ep15_bs16_seed42/
```

Generate the analysis report afterward:

```bash
python dimsum_report.py --run_dir runs/mtl_crf_microsoft__deberta-v3-small_lr2e-05_ep15_bs16_seed42
```

### Gappy-MWE experiment: B, B+c, G

Three conditions share the encoder, data split, seed, optimizer and supersense head;
only the MWE output differs.

| Condition | Flags | MWE output |
|---|---|---|
| B | `--architecture mtl_crf` | CRF over the 6 DiMSUM tags (existing model) |
| B+c | `--architecture mtl_crf --constrained_decoding` | Same CRF; Viterbi restricted to valid tag sequences |
| G | `--architecture mtl_parent` | Parent selection: each word scores every preceding word as its MWE parent (DiMSUM column 6) or "none"; chains are decoded under DiMSUM's structural constraints and converted back to tags |

Run all three with the same settings (30 epochs; the best epoch is chosen on dev).
Each command is on one line so it works in bash and PowerShell:

```bash
python dimsum_unified.py --data_dir ./dimsum-data --eval_file ./dimsum-data/scripts/dimsumeval.py --model_name microsoft/deberta-v3-small --epochs 30 --batch_size 16 --lr 2e-5 --mwe_loss_weight 3 --sup_loss_weight 2 --seed 42 --architecture mtl_crf
python dimsum_unified.py --data_dir ./dimsum-data --eval_file ./dimsum-data/scripts/dimsumeval.py --model_name microsoft/deberta-v3-small --epochs 30 --batch_size 16 --lr 2e-5 --mwe_loss_weight 3 --sup_loss_weight 2 --seed 42 --architecture mtl_crf --constrained_decoding
python dimsum_unified.py --data_dir ./dimsum-data --eval_file ./dimsum-data/scripts/dimsumeval.py --model_name microsoft/deberta-v3-small --epochs 30 --batch_size 16 --lr 2e-5 --mwe_loss_weight 3 --sup_loss_weight 2 --seed 42 --architecture mtl_parent
```

With Docker, prefix each command with `docker compose run --rm dimsum`, e.g.
`docker compose run --rm dimsum python test_decoding.py`.

Run folders: `runs/mtl_crf_microsoft__deberta-v3-small_lr2e-05_ep30_bs16_seed42/`,
`runs/mtl_crf_cd_..._seed42/` (B+c) and `runs/mtl_parent_..._seed42/` (G).

Before the full runs, `python test_decoding.py` (CPU, about a minute, needs `torch` and
`pytorch-crf`) checks the decoders and the group scorer against the gold data.

What changed in training and evaluation:

- **Dev checkpoint selection (all conditions).** After every epoch the dev split is
  scored with the official evaluator. The checkpoint with the highest dev combined F is
  kept (ties: higher dev supersense F, then the earlier epoch), and only that checkpoint
  is evaluated on test. Per-epoch dev scores are in `loss_history.csv` and
  `summary.json`. `--no_dev_selection` restores the old behavior (final epoch).
- **Exact-group scores.** `group_eval.json` (from `gappy_eval.py`) gives gappy and
  contiguous MWE precision/recall/F1 with raw TP/FP/FN, counting a predicted group as
  correct only if its token set exactly matches a gold group. It can also be run on any
  existing prediction file:
  `python gappy_eval.py dimsum-data/dimsum16.test runs/<run>/predictions.pred`
- **Diagnostics in `summary.json`.** `invalid_mwe_sentences_before_repair` counts test
  sentences whose raw MWE tags were invalid before `clean_mwe_tags` repaired them.
  For G, `parent_decoding` summarizes unconstrained vs final links, cross-gap links,
  the "none" rate, and links removed by each constraint; `parent_decode_log.jsonl`
  has the per-sentence detail (unconstrained parents, final parents, removed links
  with score margins).
- **Prediction writer.** Tokens beyond `--max_len` (truncated sentences) are now
  written as unpredicted instead of copying the gold columns. With DeBERTa-v3 at
  `--max_len 128` no DiMSUM sentence is truncated, so earlier results are unaffected.

For Docker, prefix that report command with `docker compose run --rm dimsum`.
Training can vary slightly across hardware even though the program uses a fixed
random seed.

The main project files are expected to be:

```text
your-project/
  dimsum_unified.py
  dimsum_report.py
  requirements.txt
  Dockerfile
  docker-compose.yml
  dimsum-data/
  runs/
```

---

## 1. Get the DiMSUM data

From the project root:

```bash
git clone https://github.com/dimsum16/dimsum-data.git
```

Expected files:

```text
dimsum-data/dimsum16.train
dimsum-data/dimsum16.test
dimsum-data/scripts/dimsumeval.py
dimsum-data/scripts/tags2sst.py
dimsum-data/scripts/sst2tags.py
```

---

## 2. Python 3 patch for official evaluator

The official DiMSUM evaluator was originally written for Python 2.7, so it needs a one-time Python 3 compatibility patch.

Run this after cloning `dimsum-data`:

```bash
python - <<'PY'
from pathlib import Path
import re

files = [
    Path("dimsum-data/scripts/dimsumeval.py"),
    Path("dimsum-data/scripts/tags2sst.py"),
    Path("dimsum-data/scripts/sst2tags.py"),
]

for p in files:
    text = p.read_text(encoding="utf-8")

    text = text.replace("from __future__ import print_function, division\n", "")
    text = text.replace("from __future__ import print_function\n", "")
    text = text.replace("from __future__ import division\n", "")
    text = text.replace("from __builtin__ import True\n", "")
    text = text.replace("from __builtin__ import True", "")

    text = text.replace("import sys, fileinput, json, StringIO", "import sys, fileinput, json, io")
    text = text.replace("import StringIO", "import io")
    text = text.replace("StringIO.StringIO(", "io.StringIO(")

    text = text.replace(".encode('utf-8')", "")
    text = text.replace('.encode("utf-8")', "")
    text = text.replace(".decode('utf-8')", "")
    text = text.replace('.decode("utf-8")', "")

    text = text.replace(
        "key=lambda (l,lN): not l.startswith(d)",
        "key=lambda item: not item[0].startswith(d)"
    )

    p.write_text(text, encoding="utf-8")
    print("basic patched", p)

p = Path("dimsum-data/scripts/dimsumeval.py")
text = p.read_text(encoding="utf-8")

text = re.sub(
    r"def mweval_sent\(sent, ggroups, pgroups, gmwetypes, pmwetypes, stats, indata=None\):\n",
    "def mweval_sent(sent, ggroups, pgroups, gmwetypes, pmwetypes, stats, indata=None):\n    sent = list(sent)\n",
    text,
    count=1,
)

text = text.replace(
    "tags = zip(*sent)[k]",
    "tags = list(zip(*sent))[k]"
)

text = text.replace(
    "ggroups1==map(sorted, form_groups(glinks1))",
    "ggroups1==list(map(sorted, form_groups(glinks1)))"
)
text = text.replace(
    "pgroups1==map(sorted, form_groups(plinks1))",
    "pgroups1==list(map(sorted, form_groups(plinks1)))"
)

text = text.replace(
    "sstpositions = set(glbls.keys()+plbls.keys())",
    "sstpositions = set(list(glbls.keys()) + list(plbls.keys()))"
)

text = re.sub(
    r"def f1\(prec, rec\):\n\s+return 2\*prec\*rec/\(prec\+rec\) if prec\+rec>0 else float\('nan'\)",
    """def f1(prec, rec):
    p = float(prec)
    r = float(rec)
    return 2*p*r/(p+r) if p+r > 0 else float('nan')""",
    text,
    count=1,
)

p.write_text(text, encoding="utf-8")
print("deep patched", p)
PY
```

Check that the evaluator compiles:

```bash
python -m py_compile \
  dimsum-data/scripts/dimsumeval.py \
  dimsum-data/scripts/tags2sst.py \
  dimsum-data/scripts/sst2tags.py
```

---

# Option A: Docker GPU sandbox

Use this when training on a local NVIDIA GPU. The goal is to avoid installing project libraries directly on the host system.

## Docker Compose

Recommended `docker-compose.yml`:

```yaml
services:
  dimsum:
    build:
      context: .
      dockerfile: Dockerfile

    volumes:
      - .:/workspace
      - hf_cache:/root/.cache/huggingface

    working_dir: /workspace

    gpus: all

    env_file:
      - .env

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - HF_HOME=/root/.cache/huggingface

    stdin_open: true
    tty: true
    command: bash

volumes:
  hf_cache:
```

Create a `.env` file in the project root:

```env
HF_TOKEN=
```

`HF_TOKEN` is optional for public models, but it helps avoid Hugging Face rate limits. It does not make training epochs faster after the model is downloaded.

## Build and enter the container

```bash
docker compose build
docker compose run --rm dimsum
```

You should now be inside `/workspace`.

## Verify GPU access

Inside the container:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

If this prints `cuda available: False`, the container is not using the GPU.

Common causes:

- Docker Desktop is not using WSL2.
- NVIDIA drivers are missing or outdated.
- `docker-compose.yml` does not include `gpus: all`.
- PyTorch inside the image is CPU-only.

## VS Code sync behavior

If the compose file contains:

```yaml
volumes:
  - .:/workspace
```

then files saved in VS Code should update inside the container immediately. No rebuild is needed for normal code changes.

To test sync:

```bash
# In VS Code / host project folder
echo hello > sync_test.txt
```

Then inside Docker:

```bash
ls sync_test.txt
cat sync_test.txt
```

If the file does not appear, you are probably running Docker from the wrong folder or the bind mount is missing.

---

# Option B: Local virtual environment

Use this for quick CPU tests, non-Docker debugging, or teammates who do not want to use Docker. For serious GPU training, Docker or Colab is recommended.

## Create venv

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick CPU smoke test

```bash
python dimsum_unified.py \
  --data_dir ./dimsum-data \
  --eval_file ./dimsum-data/scripts/dimsumeval.py \
  --quick_cpu
```

This is only a small sanity check. It is not the final experiment.

---

# Training commands

## RoBERTa multitask + CRF

Good first GPU run:

```bash
python dimsum_unified.py \
  --data_dir ./dimsum-data \
  --eval_file ./dimsum-data/scripts/dimsumeval.py \
  --model_name roberta-base \
  --architecture mtl_crf \
  --epochs 3 \
  --batch_size 16
```

## DeBERTa-v3-small multitask + CRF

Use a smaller batch size:

```bash
python dimsum_unified.py \
  --data_dir ./dimsum-data \
  --eval_file ./dimsum-data/scripts/dimsumeval.py \
  --model_name microsoft/deberta-v3-small \
  --architecture mtl_crf \
  --epochs 3 \
  --batch_size 8
```

If CUDA memory fails, use:

```bash
--batch_size 4
```

## Linear baseline

```bash
python dimsum_unified.py \
  --data_dir ./dimsum-data \
  --eval_file ./dimsum-data/scripts/dimsumeval.py \
  --model_name bert-base-cased \
  --architecture linear \
  --epochs 3 \
  --batch_size 16
```

---

# Recommended model order

Run models in this order:

```text
bert-base-cased + linear
bert-base-cased + mtl_crf
roberta-base + linear
roberta-base + mtl_crf
microsoft/deberta-v3-small + mtl_crf
microsoft/deberta-v3-base + mtl_crf, only if GPU memory allows
```

For the project writeup, compare models using the official DiMSUM scores, not only internal macro F1.

---

# Report generation

After a training run finishes, generate the report for the latest run:

```bash
python dimsum_report.py
```

This auto-detects the latest folder under `runs/` and creates:

```text
runs/<latest_run>/report/
  summary_report.md
  sentence_errors.md
  token_errors.csv
  supersense_confusions.csv
  mwe_tag_confusions.csv
  official_scores.png
  error_breakdown.png
  supersense_top_confusions.png
  mwe_tag_confusion.png
```

Generate a report for a specific run:

```bash
python dimsum_report.py --run_dir runs/mtl_crf_roberta-base_lr2e-05_ep3_bs16
```

Compare all runs:

```bash
python dimsum_report.py --all_runs
```

This creates:

```text
reports/all_runs/runs_summary.csv
reports/all_runs/runs_f1_comparison.png
```

---

# Output folders

Each training run creates a folder like:

```text
runs/mtl_crf_roberta-base_lr2e-05_ep3_bs16/
  model.pt
  predictions.pred
  summary.json
  official_eval.txt
  report/
```

Important files:

- `summary.json`: internal training summary
- `official_eval.txt`: official DiMSUM evaluator output
- `predictions.pred`: model predictions
- `report/summary_report.md`: readable project summary
- `report/sentence_errors.md`: qualitative examples for linguistic analysis
- `report/token_errors.csv`: spreadsheet-friendly token-level errors

---

# Common issues

## `Lambda expression parameters cannot be parenthesized`

The evaluator is still Python 2 style. Run the Python 3 evaluator patch section again.

## `No module named StringIO`

The evaluator patch did not fully apply. Run the Python 3 evaluator patch section again.

## `zip object is not subscriptable`

The evaluator patch did not fully apply. Run the Python 3 evaluator patch section again.

## `Ratio > int` TypeError

The evaluator patch did not fully apply. Run the Python 3 evaluator patch section again.

## `torch.load` vulnerability / torch >= 2.6 required

This happens when Transformers tries to load old `.bin` model weights with an older PyTorch version. The code should load models with:

```python
AutoModel.from_pretrained(model_name, use_safetensors=True).float()
```

If the error persists, update PyTorch in the Docker image or virtual environment.

## `mat1 and mat2 must have the same dtype, but got Half and Float`

The encoder output and classifier heads are using different numeric precision. The model code should include:

```python
self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
out = out.to(self.mwe_head.weight.dtype)
```

## Docker sees CPU only

Check:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

If it says CPU only, verify Docker GPU setup and CUDA-compatible PyTorch.

---

# Minimal workflow summary

Docker GPU workflow:

```bash
docker compose build
docker compose run --rm dimsum

git clone https://github.com/dimsum16/dimsum-data.git
# run evaluator patch once

python dimsum_unified.py \
  --data_dir ./dimsum-data \
  --eval_file ./dimsum-data/scripts/dimsumeval.py \
  --model_name roberta-base \
  --architecture mtl_crf \
  --epochs 3 \
  --batch_size 16

python dimsum_report.py
```

Venv workflow:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or .\.venv\Scripts\Activate.ps1 on Windows PowerShell

pip install -r requirements.txt
git clone https://github.com/dimsum16/dimsum-data.git
# run evaluator patch once

python dimsum_unified.py --data_dir ./dimsum-data --eval_file ./dimsum-data/scripts/dimsumeval.py --quick_cpu
python dimsum_report.py
```
