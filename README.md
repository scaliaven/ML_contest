# ML_contest — Audio Classification Pipeline

A deep-learning pipeline for a **4-class audio classification** contest. Raw audio
clips are separated into vocals, converted into mel-spectrograms, and classified
with an ensemble of convolutional / attention networks. The pipeline covers the
full path from raw `.mp3` files to a submission `.csv`.

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

## Models

`main.py` selects an architecture via the `name` variable. Available options:

- `resnet` — ResNet-18 (`model_zoo/resnet18.py`)
- `resnext50` / `resnext101` — torchvision ResNeXt (single-channel input, `num_classes=4`)
- `densenet` — torchvision DenseNet-201 (single-channel input)
- `shake_resnet` / `shake_resnext` — Shake-Shake regularized nets
- `attention_resnet`, `attention_next_56`, `attention_next_92` — Residual Attention Networks
- `lstm` — recurrent baseline (`model_zoo/Transformer.py`)

**Training details:** cross-entropy loss, SGD with momentum + weight decay, cosine-annealing
warm-restart schedule, gradient-value clipping, and checkpointing on best validation accuracy.
Regularization uses **mixup** (first half of training) and **SpecAugment** (time/frequency
masking via `torchaudio`).

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

All packages in `requirements.txt` and `requirements_spleeter.txt` are **intentionally
pinned to the exact versions used for the contest** so results stay reproducible. They
are deliberately *not* upgraded to newer releases.

Automated dependency updates (**Dependabot**) have been **removed / disabled** for this
repository — no `.github/dependabot.yml` is present, and automated update / security
pull requests are not wanted here. Any Dependabot security alerts reported against these
frozen versions are expected and are left as-is by design. (Fully turning off the
alerts themselves is a repository *Settings → Code security* action.)

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

## License

See [`LICENSE`](LICENSE).
