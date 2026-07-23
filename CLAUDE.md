# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this project is

A deep-learning pipeline for a **4-class music audio classification** Kaggle contest. Each
track's **vocal** stem (Spleeter) is turned into a `[1, 128, 128]` grayscale mel-spectrogram
(Librosa) and classified with CNN / attention networks, then combined via ensembling. The
final submission (ResNeXt-50 + DenseNet-201 soft-voting ensemble) scored **83.04%**. See
[`README.md`](README.md) for the pipeline diagram and per-file table, and
[`docs/ML_project.pdf`](docs/ML_project.pdf) for the full write-up (methodology, experiments,
results).

## Architecture at a glance

```
audio (.mp3) → voice_split.py (Spleeter 2stems) → move_split.py
            → data_processing/gray_scale.py (mel-spectrogram → .pt)
            → data_util.py (CustomImageDataset) → main.py (train)
            → predict.py / naive_ensumble.py / meta_model.py → my_submission.csv
```

Key modules:
- **`main.py`** — training/inference driver. An architecture is chosen via the `name`
  string, and behavior is controlled by a flag block at the top of the file
  (`save`, `predict`, `continue_train`, `epochs`, `name`, `PATH`).
- **`data_util.py`** — `Dataset` classes. Spectrograms are loaded from `.pt` files and
  cropped with `[:, :, 1:129]`; `CustomImageDataset` can oversample class `0` (`balance=True`).
- **`model_zoo/`** — model definitions (resnet18, shake-shake variants, residual
  attention networks, LSTM/Transformer). torchvision models (ResNeXt, DenseNet) are
  built inline in `main.py`/`meta_model.py`/`naive_ensumble.py`.
- **`predict.py`** — single-model inference → CSV. `naive_ensumble.py` — soft-voting.
  `meta_model.py` — stacked meta-model over base models.

## Environments

There are **two separate Python environments** — do not merge them:
- `requirements.txt` — PyTorch training/inference stack.
- `requirements_spleeter.txt` — Spleeter preprocessing. Spleeter 2.4.0 pins an
  end-of-life TensorFlow 2.9 line, incompatible with the PyTorch env.

## Running

```bash
python main.py            # train/infer (edit the flag block first)
sbatch run_main.sh        # SLURM (NYU HPC H100 partition)
python predict.py         # inference → my_submission.csv
```

There is no test suite, linter config, or CI in this repo.

## Conventions & gotchas (read before editing)

- **Hard-coded absolute paths.** Scripts reference `/scratch/hh3043/ML_contest/...`
  (an HPC scratch dir) for datasets, checkpoints, and the output CSV. Do not assume
  these exist locally; when adding code, keep paths configurable rather than hard-coding
  new ones where practical.
- **Config is edit-in-place**, not CLI args. Runtime options live in the flag block at
  the top of `main.py` / `meta_model.py`. Preserve that style unless asked to refactor.
- **`num_classes=4` everywhere.** Single-channel (grayscale spectrogram) input. The
  torchvision stems are deliberately patched for the small `128×128` input: the first conv
  becomes `3×3` stride-1 padding-1 and the following max-pool is skipped (kernel 1) — this
  avoids over-downsampling and is intentional (see the report), so keep it when swapping
  architectures.
- **Dependencies are pinned to the original contest versions** for reproducibility and are
  intentionally not upgraded. Earlier Dependabot bumps (`certifi`, `setuptools`, `keras`)
  were reverted back to those original pins. Don't upgrade pins, and don't re-add
  `.github/dependabot.yml` — automated update PRs are not used.
- Filenames follow the existing (sometimes misspelled) names, e.g. `naive_ensumble.py`,
  `voice_split.py`. Match existing style; don't rename files without being asked.

## Git

- Do not commit or push unless asked.
- Do not re-add `.github/dependabot.yml`; keep dependency versions at their original pins.
