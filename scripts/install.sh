#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

env_name="${CONDA_ENV:-scal3r}"
python_version="${PYTHON_VERSION:-3.10}"
use_uv="${USE_UV:-1}"
torch_channel="${TORCH_CHANNEL:-auto}"

log() {
    echo "[install] $*"
}

die() {
    echo "[install] $*" >&2
    exit 1
}

ensure_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        die "conda was not found in PATH."
    fi

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
}

ensure_env() {
    if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
        log "Using existing conda environment: ${env_name}"
    else
        log "Creating conda environment ${env_name} with Python ${python_version}"
        conda create -n "$env_name" "python=${python_version}" -y
    fi
    conda activate "$env_name"
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    log "uv not found; installing it with the official installer."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "Neither curl nor wget is available. Install uv manually or rerun with USE_UV=0."
    fi

    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        die "uv was installed but is still not on PATH. Please add ~/.local/bin to PATH and rerun."
    fi
}

resolve_torch_channel() {
    local uname_s
    uname_s="$(uname -s)"

    if [[ "$uname_s" == "Darwin" ]]; then
        echo "macos"
        return
    fi

    if [[ "$torch_channel" != "auto" ]]; then
        echo "$torch_channel"
        return
    fi

    if command -v nvcc >/dev/null 2>&1; then
        local cuda_version
        cuda_version="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
        case "$cuda_version" in
            12.8|12.9|13.*) echo "cu128" ;;
            12.*) echo "cu126" ;;
            11.*) echo "cu118" ;;
            *) echo "cpu" ;;
        esac
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local smi_cuda_version
        smi_cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
        case "$smi_cuda_version" in
            12.8|12.9|13.*) echo "cu128" ;;
            12.*) echo "cu126" ;;
            11.*) echo "cu118" ;;
            *) echo "cpu" ;;
        esac
        return
    fi

    echo "cpu"
}

install_torch_with_uv() {
    local resolved_channel="$1"

    if [[ "$resolved_channel" == "macos" ]]; then
        log "Installing PyTorch for macOS"
        uv pip install torch torchvision
        return
    fi

    if [[ "$resolved_channel" == "cpu" ]]; then
        log "Installing CPU PyTorch wheels"
    else
        log "Installing PyTorch wheels from the ${resolved_channel} index"
    fi
    uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/${resolved_channel}"
}

install_torch_with_pip() {
    local resolved_channel="$1"

    if [[ "$resolved_channel" == "macos" ]]; then
        log "Installing PyTorch for macOS"
        python -m pip install torch torchvision
        return
    fi

    if [[ "$resolved_channel" == "cpu" ]]; then
        log "Installing CPU PyTorch wheels"
    else
        log "Installing PyTorch wheels from the ${resolved_channel} index"
    fi
    python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${resolved_channel}"
}

install_with_uv() {
    ensure_uv
    install_torch_with_uv "$(resolve_torch_channel)"
    log "Installing Python requirements with uv pip"
    uv pip install -r requirements.txt
    uv pip install -e .
}

install_with_pip() {
    python -m pip install --upgrade pip setuptools wheel
    install_torch_with_pip "$(resolve_torch_channel)"
    log "Installing Python requirements with pip"
    python -m pip install -r requirements.txt
    python -m pip install -e .
}

ensure_conda
ensure_env

if [[ "$use_uv" == "1" ]]; then
    install_with_uv
else
    install_with_pip
fi

echo
log "Installation finished."
log "Activate the environment with: conda activate ${env_name}"
log "Verify with: python -m scal3r.run --help"
