# Installation

This page expands the short installation block in the project README.

## Recommended path

The recommended setup is:

```bash
bash scripts/install.sh
```

By default, the script:

1. Creates or reuses a conda environment named `scal3r`.
2. Uses Python 3.10 by default.
3. Installs `uv` automatically if it is not already available.
4. Installs PyTorch separately using the official PyTorch wheels for your platform.
5. Installs `requirements.txt`.
6. Installs Scal3R itself in editable mode with `-e .`.

After the script finishes:

```bash
conda activate scal3r
python -m scal3r.run --help
python -m scal3r.pipelines.backend --help
```

Download the two required checkpoints to `data/checkpoints/`:

```bash
mkdir -p data/checkpoints && hf download xbillowy/Scal3R scal3r.pt --repo-type model --local-dir data/checkpoints
mkdir -p data/checkpoints && curl -L https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt -o data/checkpoints/dino_salad.ckpt
```

## Manual installation

If you want to create the conda environment yourself:

```bash
conda create -n scal3r python=3.10 -y
conda activate scal3r
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # choose your own CUDA version
uv pip install -r requirements.txt
uv pip install -e .
```

If you need to override the default PyTorch wheel channel or the conda environment name:

```bash
CONDA_ENV=scal3r-cu126 bash scripts/install.sh
TORCH_CHANNEL=cu118 bash scripts/install.sh
TORCH_CHANNEL=cu126 bash scripts/install.sh
TORCH_CHANNEL=cu128 bash scripts/install.sh
TORCH_CHANNEL=cpu bash scripts/install.sh
```

If you prefer standard `pip`, or if `uv` causes trouble on your machine, you can also opt out of `uv`:

```bash
USE_UV=0 bash scripts/install.sh
```

## Troubleshooting

### `uv` is installed but not found

Add `~/.local/bin` to your shell `PATH` and rerun the install script.

### I want plain `pip`

Use:

```bash
USE_UV=0 bash scripts/install.sh
```

### I want a different conda environment name

Use:

```bash
CONDA_ENV=scal3r-dev bash scripts/install.sh
```
