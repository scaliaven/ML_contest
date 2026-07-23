# ML_contest — Music Audio Classification Pipeline

A deep-learning pipeline for a **4-class music audio classification** Kaggle contest.
Each track is source-separated into its vocal part, converted into a mel-spectrogram,
and treated as a single-channel (grayscale) image for classification by an ensemble of
convolutional / attention networks. The pipeline covers the full path from raw `.mp3`
files to a submission `.csv`.

> **Result:** the final submission — a soft-voting ensemble of **ResNeXt-50 + DenseNet-201** —
> reached **83.04%** test accuracy. Full write-up: [`docs/ML_project.pdf`](docs/ML_project.pdf).

## Pipeline overview

```
.mp3 clips
   │  voice_split.py        (Spleeter 2-stems source separation)
   ▼
vocals.wav
   │  move_split.py         (collect vocals into a flat directory)
   ▼
train_mp3s / test_mp3s
   │  data_processing/gray_scale.py   (mel-spectrogram → .pt tensor)
   │  data_processing/convert.py      (mel-spectrogram → .png image, alt. path)
   ▼
spectrogram tensors (.pt)
   │  data_util.py          (CustomImageDataset: load, crop, balance, normalize)
   ▼
   │  main.py               (train a CNN / attention model)
   ▼
checkpoint .pth
   │  predict.py            (single-model inference)
   │  naive_ensumble.py     (soft-voting ensemble)
   │  meta_model.py         (stacked meta-model)
   ▼
my_submission.csv
```

## Repository layout

| Path | Purpose |
|------|---------|
| `voice_split.py` | Runs Spleeter (`2stems`) to split each clip into vocals / accompaniment. |
| `move_split.py` | Moves the separated `vocals.wav` files into flat train/test folders. |
| `data_processing/gray_scale.py` | Converts each `.wav` into a mel-spectrogram tensor saved as `.pt` (training input). |
| `data_processing/convert.py` | Alternative: renders mel-spectrograms to `.png` images. |
| `data_processing/data_spliting.py`, `permutate.py` | Train/val splitting and data-permutation helpers. |
| `data_util.py` | `Dataset` classes: spectrogram loading, cropping (`[:, :, 1:129]`), optional class balancing, and the meta-model datasets. |
| `main.py` | Main training/inference driver. Selects a model, applies augmentation, trains, and writes predictions. |
| `model_zoo/` | Model definitions: `resnet18`, `shake_resnet`, `shake_resnext`, `residual_attention_network`, `Transformer` (LSTM/attention), and building blocks. |
| `GAN.py`, `autoencoder.py` | Auxiliary / experimental generative models. |
| `predict.py` | Single-model inference → `my_submission.csv`. |
| `naive_ensumble.py` | Soft-voting ensemble (sums logits of ResNeXt-50 + DenseNet-201). |
| `meta_model.py` | Stacked meta-model trained on top of base-model outputs. |
| `run_main.sh` | SLURM batch script (configured for an NYU HPC H100 partition). |
| `requirements.txt` | PyTorch training/inference environment. |
| `requirements_spleeter.txt` | Separate Spleeter/TensorFlow environment for preprocessing (see note below). |
| `docs/ML_project.pdf` | The project report (methodology, experiments, and results). |

## Models

`main.py` selects an architecture via the `name` variable. Available options:

- `resnet` — ResNet-18 (`model_zoo/resnet18.py`)
- `resnext50` / `resnext101` — torchvision ResNeXt (single-channel input, `num_classes=4`)
- `densenet` — torchvision DenseNet-201 (single-channel input)
- `shake_resnet` / `shake_resnext` — Shake-Shake regularized nets
- `attention_resnet`, `attention_next_56`, `attention_next_92` — Residual Attention Networks
- `lstm` — recurrent baseline (`model_zoo/Transformer.py`)

**Architecture modifications.** The inputs are `[1, 128, 128]` grayscale spectrograms, not
`224×224` photos, so the aggressive early downsampling of the standard torchvision stems is
removed: the first convolution is changed to a `3×3` filter with stride 1 / padding 1, and the
following max-pool is skipped (kernel 1, stride 1, padding 0). All heads use `num_classes=4`.

**Training details.** Cross-entropy loss, SGD with momentum + weight decay, a cosine-annealing
(warm-restart) learning-rate schedule, gradient-value clipping, and checkpointing on best
validation accuracy. Regularization uses **Mixup** for the first half of training (then plain
cross-entropy — Mixup over too many epochs hurt generalization on deep nets like DenseNet-201)
and **SpecAugment** (per-batch time/frequency masking via `torchaudio`).

## Data representation

Spleeter (`2stems`) splits each track into vocal / instrumental parts; the **vocal** stem is
kept. Librosa turns it into a mel-spectrogram of shape `[128, 130]` (frequency × time), which
is sliced to `[128, 128]` and reshaped to `[1, 128, 128]` so it can be fed to image models as a
single-channel image.

## Experiments & findings

From the project report ([`docs/ML_project.pdf`](docs/ML_project.pdf)):

- **Grayscale beats RGB** — using single-channel spectrogram images yielded higher accuracy
  than RGB renderings.
- **Mixup + SpecAugment ≈ +2%** accuracy over no augmentation.
- **Unweighted ensembles help; weighted ones overfit** — summing model output probabilities
  and taking the argmax improved results, while a learned/reweighted meta-model overfit and
  slightly reduced test accuracy. The best combination was **DenseNet-201 + ResNeXt-50**.
- **Not finished / future work** — GAN- and Autoencoder-based data augmentation were started
  but not completed; K-fold cross-validation would fit the data better, since tracks are split
  into train/test first and then sliced into 3-second snippets (so train/val snippets come from
  the same songs while test songs are unseen).

## Setup

The training and preprocessing steps use **two separate environments**, because Spleeter
pins an older TensorFlow that is incompatible with the PyTorch stack.

**1. Training / inference (PyTorch):**

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.9+
pip install -r requirements.txt
```

**2. Preprocessing (Spleeter, only needed to regenerate features):**

```bash
python -m venv .venv-spleeter && source .venv-spleeter/bin/activate
pip install -r requirements_spleeter.txt
```

> Spleeter 2.4.0 constrains this second environment to an end-of-life TensorFlow 2.9
> line, so some of its pinned transitive dependencies cannot be upgraded without
> replacing Spleeter. It is kept isolated from the main environment for that reason.

### Dependency versions & Dependabot

All packages in `requirements.txt` and `requirements_spleeter.txt` are pinned to the
**original versions used for the contest** so results stay reproducible; they are not
upgraded. Earlier automated **Dependabot** bumps (`certifi`, `setuptools`, `keras`) have
been reverted back to those original pins.

Dependabot is **not used** for this repository — there is no `.github/dependabot.yml`, so
version-update pull requests are disabled. (Turning off Dependabot *alerts* as well is a
repository *Settings → Code security* action.)

## Usage

### 1. Preprocess (Spleeter environment)

```bash
python voice_split.py                    # separate vocals from each clip
python move_split.py                     # collect vocals.wav into flat folders
python data_processing/gray_scale.py     # mel-spectrogram tensors (.pt)
```

### 2. Train (PyTorch environment)

Configuration lives in the flag block near the top of `main.py`:

```python
save            = False   # True on first run: build & cache the .pt datasets
predict         = False   # True to run inference only from a checkpoint
continue_train  = False   # resume from an existing checkpoint
epochs          = 20
name            = "resnext101"   # architecture to train (see Models)
PATH            = ".../checkpoint_resnext101_aug_test_1.pth"
```

Then run:

```bash
python main.py
# or submit to SLURM:
sbatch run_main.sh
```

### 3. Predict / ensemble

```bash
python predict.py           # single model → my_submission.csv
python naive_ensumble.py    # ResNeXt-50 + DenseNet-201 soft-voting
python meta_model.py        # stacked meta-model
```

## Data paths

⚠️ The scripts currently use **absolute paths hard-coded** to an HPC scratch directory
(e.g. `/scratch/hh3043/ML_contest/...`). Before running elsewhere, update these paths
(dataset directories, checkpoint paths, and the output `my_submission.csv` location) to
match your environment.

## Report & compute

The full methodology, experiments, and results are documented in
[`docs/ML_project.pdf`](docs/ML_project.pdf) (author: Hongjia Huang, NYU).

All models train on an RTX 8000; deeper models (DenseNet-201, ResNeXt-50) benefit from
A100 / H100 / A800-class GPUs. Experiments were run on the NYU Greene and NYU Shanghai HPC
clusters (see `run_main.sh` for the SLURM configuration).

## Contributing / working in this repo

Contributor and agent guidance — architecture notes, conventions, and gotchas
(hard-coded paths, the edit-in-place config style, and the frozen-dependency /
no-Dependabot policy) — lives in [`CLAUDE.md`](CLAUDE.md).

## License

See [`LICENSE`](LICENSE).
